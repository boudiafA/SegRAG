from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from segrag import __version__
from segrag.modeling.feature_matching import DualThresholdPromptGenerator, _label_components
from segrag.modeling.iccd import _apply_adaptive_top_k_features, run_filter_from_scored_bank
from segrag.modeling.sam3_text_points import Sam3TextAndPointsEvaluator
from segrag.reproducibility.paper_benchmark import _validate_resume_metadata
from segrag.stages.filter_bank import METHODS
from segrag.utils import checkpoints


ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_metadata():
    metadata = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', metadata, re.MULTILINE).group(1) == __version__


@pytest.mark.parametrize("script", sorted((ROOT / "scripts").glob("*.py")), ids=lambda path: path.stem)
def test_source_checkout_cli_help(script, tmp_path):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_all_local_imports_resolve_after_cleanup():
    for path in (ROOT / "src" / "segrag").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "segrag" or node.module.startswith("segrag."):
                    target = ROOT / "src" / Path(*node.module.split("."))
                    assert target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), (path, node.module)


def test_discarded_paths_are_not_shipped():
    for name in ("build_bank_v2", "build_bank_v3", "merge_masks", "merge_masks_impl", "run_pipeline"):
        assert not (ROOT / "src" / "segrag" / "stages" / f"{name}.py").exists()
    assert METHODS == ("fixed", "adaptive_q75")


@pytest.mark.parametrize("field", ["workload", "fingerprints"])
def test_resume_rejects_changed_workload_or_code(tmp_path, field):
    metadata = {
        "protocol_version": "segrag-table4-v1", "dataset": "PC-59", "shot": 5,
        "fingerprints": {"segrag_source": "previous"},
        "workload": {"max_images": 1, "save_mask_json": True},
    }
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps(metadata))
    _validate_resume_metadata(str(path), metadata)
    changed = dict(metadata)
    changed[field] = {"changed": True}
    with pytest.raises(ValueError, match=field):
        _validate_resume_metadata(str(path), changed)


def test_adaptive_q75_is_a_cutoff_not_top_25_percent():
    scores = torch.tensor([0.60, 0.70, 0.75, 0.80, 0.90, 1.0])
    keep, q75, threshold, survivors = _apply_adaptive_top_k_features(scores, 10000)
    assert q75 == pytest.approx(0.875)
    assert threshold == pytest.approx(0.7875)
    assert keep.tolist() == [False, False, False, True, True, True]
    assert survivors == 3


def test_scored_bank_filter_and_resume(tmp_path):
    source = tmp_path / "scored" / "crop"
    source.mkdir(parents=True)
    features = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    torch.save(features, source / "1_1.pt")
    np.save(source / "1_1.scores.npy", np.array([0.60, 0.70, 0.75, 0.80, 0.90, 1.0], dtype=np.float32))
    kwargs = dict(input_dir=str(source.parent), output_dir=str(tmp_path / "filtered"),
                  method="adaptive_q75", keep_threshold=None, top_k_features=2)
    report = run_filter_from_scored_bank(**kwargs, resume=False)
    output = torch.load(tmp_path / "filtered" / "crop" / "1_1.pt", weights_only=True)
    assert torch.equal(output, features[-2:])
    assert report["summary"]["total_features_out"] == 2
    resumed = run_filter_from_scored_bank(**kwargs, resume=True)
    assert resumed["summary"]["total_features_out"] == 2
    assert resumed["summary"]["classes_skipped"] == 1


def test_hybrid_tsg_is_uncapped_and_maps_patch_centers():
    generator = object.__new__(DualThresholdPromptGenerator)
    similarity = np.zeros((96, 96), dtype=np.float32)
    for row in range(0, 96, 24):
        for col in range(0, 96, 24):
            similarity[row:row + 2, col:col + 2] = 0.85
            similarity[row, col] = 0.95
    prompts = generator.generate_prompts_from_sim_map(
        similarity, original_size=(192, 288), loose_threshold=0.8,
        min_peak_distance=10, min_component_size=4,
    )
    assert len(prompts) == 16
    for prompt in prompts:
        assert prompt["x"] == pytest.approx((prompt["col"] + 0.5) * 3)
        assert prompt["y"] == pytest.approx((prompt["row"] + 0.5) * 2)
    assert not generator.generate_prompts_from_sim_map(
        similarity, (192, 288), 0.8, 10, 5,
    )


def test_components_use_eight_connectivity():
    _, count = _label_components(np.eye(4, dtype=bool))
    assert count == 1


def test_joint_prompt_uses_one_grounding_call_and_preserves_image_state():
    calls = []

    class Prompt:
        def append_points(self, points, labels):
            self.points, self.labels = points, labels

    class Processor:
        def reset_all_prompts(self, state):
            calls.append("reset")

        def _forward_grounding(self, state):
            calls.append("joint")
            assert "text" in state["backbone_out"]
            prompt = state["geometric_prompt"]
            assert torch.equal(prompt.points, torch.tensor([[[0.5, 0.5]]]))
            assert prompt.labels.shape == (1, 1)
            return {**state, "masks": np.ones((1, 4, 6)), "scores": np.array([0.9])}

        def set_text_prompt(self, **kwargs):
            raise AssertionError("Joint prompting must not decode text first")

    evaluator = object.__new__(Sam3TextAndPointsEvaluator)
    evaluator.device = "cpu"
    evaluator.model = SimpleNamespace(
        backbone=SimpleNamespace(forward_text=lambda *_args, **_kwargs: {"text": torch.ones(1)}),
        _get_dummy_prompt=Prompt,
    )
    evaluator.processor = Processor()
    state = {"backbone_out": {"image": torch.ones(1)}}
    masks, _, status = evaluator.predict_text_and_points(
        state, "a crop", np.array([[3, 2]], dtype=np.float32), np.array([1]), (4, 6),
    )
    assert calls == ["reset", "joint"]
    assert "text" not in state["backbone_out"]
    assert "geometric_prompt" not in state
    assert len(masks) == 1 and masks[0].all()
    assert not status["used_text_only_fallback"]


def test_zero_points_use_text_fallback():
    calls = []
    evaluator = object.__new__(Sam3TextAndPointsEvaluator)
    evaluator.device = "cpu"
    evaluator.processor = SimpleNamespace(
        reset_all_prompts=lambda _state: None,
        set_text_prompt=lambda **kwargs: calls.append(kwargs["prompt"]) or {},
    )
    masks, _, status = evaluator.predict_text_and_points(
        {}, "a crop", np.empty((0, 2)), np.empty(0), (4, 6),
    )
    assert calls == ["a crop"] and masks == []
    assert status["used_text_only_fallback"]
    assert not status["combined_prediction_error"]


def test_checkpoint_pin_matches_archived_provenance():
    archived = json.loads((ROOT / "reproducibility" / "software.json").read_text())["sam3"]
    assert checkpoints.SAM3_REVISION == archived["huggingface_revision"]
    assert checkpoints.SAM3_SHA256 == archived["checkpoint_sha256"]


def test_local_checkpoint_is_checked_without_download(tmp_path, monkeypatch):
    path = tmp_path / "sam3.pt"
    path.write_bytes(b"not the official weights")
    monkeypatch.setenv("SEGRAG_SAM3_CHECKPOINT", str(path))
    with pytest.raises(ValueError, match="hash mismatch"):
        checkpoints.resolve_sam3_checkpoint()
    monkeypatch.setattr(checkpoints, "SAM3_SHA256", checkpoints.sha256_file(path))
    assert checkpoints.resolve_sam3_checkpoint() == str(path)
