from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from segrag.reproducibility.protocol import (
    PAPER_PROTOCOL_VERSION,
    build_exact_support_annotations,
    load_json,
    load_paper_config,
    selected_supports,
    validate_query_manifest,
    validate_support_manifest,
)
from segrag.reproducibility.paper_benchmark import _prepare_workspace, _score_verification
from segrag.modeling.iccd import (
    _shared_support_image_ids,
    _single_reference_keep_mask,
    score_class,
)
from segrag.pipelines.main import DatasetLayout
from segrag.stages.build_bank import _validate_support_image_limit


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "pc59": ("PC-59", 59),
    "cityscapes": ("Cityscapes", 19),
    "ade20k150": ("ADE20K_150", 150),
    "lvis": ("LVIS", 1203),
}


@pytest.mark.parametrize("key,expected", DATASETS.items())
def test_frozen_support_protocols_are_nested_and_complete(key, expected):
    dataset, one_shot_classes = expected
    manifest = load_json(ROOT / "splits" / "standard" / key / "supports_1_5shot.json")
    one = selected_supports(manifest, 1)
    five = selected_supports(manifest, 5)

    one_validation = validate_support_manifest(manifest, dataset, 1)
    five_validation = validate_support_manifest(manifest, dataset, 5)
    one_keys = {(int(row["class_id"]), int(row["image_id"])) for row in one}
    five_rank1 = {
        (int(row["class_id"]), int(row["image_id"]))
        for row in five
        if int(row["rank_within_class"]) == 1
    }

    assert manifest["protocol_version"] == PAPER_PROTOCOL_VERSION
    assert one_validation["classes"] == one_shot_classes
    assert one_validation["items"] == one_shot_classes
    assert five_validation["items"] == five_validation["classes"] * 5
    assert five_rank1.issubset(one_keys)


@pytest.mark.parametrize(
    "config_name,dataset",
    [
        ("pc59.yaml", "PC-59"),
        ("cityscapes.yaml", "Cityscapes"),
        ("ade20k150.yaml", "ADE20K_150"),
        ("lvis.yaml", "LVIS"),
    ],
)
def test_paper_configs_encode_score_producing_parameters(config_name, dataset):
    config = load_paper_config(ROOT / "configs" / "paper" / config_name)
    assert config["dataset"] == dataset
    assert config["bank"]["foreground_occupancy_threshold"] == 0.90
    assert config["bank"]["top_k_features"] == 10000
    assert config["tsg"]["similarity_threshold"] == 0.80
    assert config["tsg"]["validation_threshold"] == 0.80
    assert config["tsg"]["min_component_size"] == 4
    assert config["tsg"]["min_peak_distance"] == 10


def test_exact_support_annotation_rewrites_reference_image_path(tmp_path):
    annotation = {
        "images": [{"id": 7, "file_name": "source/train.jpg", "width": 4, "height": 4}],
        "annotations": [
            {"id": 11, "image_id": 7, "category_id": 3, "segmentation": [], "bbox": [], "area": 1},
            {"id": 12, "image_id": 7, "category_id": 4, "segmentation": [], "bbox": [], "area": 1},
        ],
        "categories": [{"id": 3, "name": "crop"}, {"id": 4, "name": "soil"}],
    }
    supports = [
        {
            "image_id": 7,
            "class_id": 3,
            "class_name": "crop",
            "image_path": "reference_imgs/fixed/train.jpg",
        }
    ]
    output = tmp_path / "support.json"
    subset = build_exact_support_annotations(annotation, supports, str(output))

    assert subset["images"] == [
        {"id": 7, "file_name": "fixed/train.jpg", "width": 4, "height": 4}
    ]
    assert [row["id"] for row in subset["annotations"]] == [11]
    assert subset["categories"] == [{"id": 3, "name": "crop"}]
    assert json.loads(output.read_text()) == subset


def test_support_validation_rejects_duplicate_rank_and_image():
    manifest = {
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "dataset": "tiny",
        "valid_classes_by_shot": {"5": [{"class_id": 1}]},
        "shot_items": {
            "5": [
                {"class_id": 1, "image_id": image_id, "rank_within_class": rank}
                for image_id, rank in [(10, 1), (10, 1), (11, 2), (12, 3), (13, 4)]
            ]
        },
    }
    with pytest.raises(ValueError, match="bad_ranks"):
        validate_support_manifest(manifest, "tiny", 5)


def test_query_validation_rejects_duplicate_pairs():
    annotation = {"annotations": [{"image_id": 2, "category_id": 3}]}
    row = {"image_id": 2, "class_id": 3}
    manifest = {
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "dataset": "tiny",
        "items": [row, dict(row)],
    }
    with pytest.raises(ValueError, match="duplicates=1"):
        validate_query_manifest(manifest, "tiny", annotation)


def test_score_verification_uses_archived_miou():
    config = {
        "reported_miou": {"5": 0.5},
        "reproduction_tolerance": 0.001,
    }
    verification = _score_verification(
        config,
        5,
        {"text_and_point": {"global": {"mIoU": 0.5005}}},
    )
    assert verification["within_tolerance"]
    assert verification["absolute_error"] == pytest.approx(0.0005)


def test_nonempty_workspace_requires_explicit_resume_or_overwrite(tmp_path):
    workspace = {"root": str(tmp_path / "paper-run")}
    Path(workspace["root"]).mkdir()
    (Path(workspace["root"]) / "old.json").write_text("{}")

    with pytest.raises(FileExistsError, match="--resume"):
        _prepare_workspace(Namespace(resume=False, overwrite=False), workspace)

    _prepare_workspace(Namespace(resume=True, overwrite=False), workspace)


def test_shared_support_ids_use_one_exact_set():
    grouped_bank = {10: [("10_1.pt", "/tmp/10_1.pt")], 20: [("20_2.pt", "/tmp/20_2.pt")]}

    assert _shared_support_image_ids(
        grouped_bank=grouped_bank,
        annotated_image_ids=[10, 20],
        support_image_limit=2,
        class_name="crop",
    ) == [10, 20]


def test_shared_support_ids_reject_different_images():
    grouped_bank = {10: [("10_1.pt", "/tmp/10_1.pt")], 20: [("20_2.pt", "/tmp/20_2.pt")]}

    with pytest.raises(RuntimeError, match="support-set mismatch"):
        _shared_support_image_ids(
            grouped_bank=grouped_bank,
            annotated_image_ids=[10, 30],
            support_image_limit=2,
            class_name="crop",
        )


def test_stage_cli_validates_one_support_limit():
    _validate_support_image_limit(30)
    _validate_support_image_limit(None)
    with pytest.raises(ValueError, match="at least 1"):
        _validate_support_image_limit(0)


def test_iccd_scores_each_source_against_other_supports_only(tmp_path, monkeypatch):
    grouped_bank = {}
    prepared_targets = []
    for image_id in (1, 2, 3):
        feature_path = tmp_path / f"{image_id}_1.pt"
        torch.save(torch.tensor([[1.0, 0.0]]), feature_path)
        grouped_bank[image_id] = [(feature_path.name, str(feature_path))]
        prepared_targets.append(
            {
                "image_id": image_id,
                "grid": torch.tensor([[1.0, 0.0]]),
                "mask": torch.tensor([True]),
            }
        )

    monkeypatch.setattr(
        "segrag.modeling.iccd.prepare_target_batches",
        lambda **_kwargs: [prepared_targets],
    )
    _, _, good_by_file, bad_by_file = score_class(
        grouped_bank=grouped_bank,
        target_worklist=[],
        model=None,
        device="cpu",
        query_chunk=8,
        sim_floor=0.0,
        target_batch_size=3,
        num_workers=0,
        keep_threshold=None,
        min_matches=1,
        target_references=None,
        early_accept=False,
        selection_mode="top-k-images",
        class_name="crop",
    )

    assert all(int(vote.item()) == 2 for vote in good_by_file.values())
    assert all(int(vote.item()) == 0 for vote in bad_by_file.values())


def test_single_reference_uses_raw_occupancy_without_within_image_scoring(tmp_path):
    feature_path = tmp_path / "1_1.pt"
    raw_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    torch.save(raw_features, feature_path)

    _, scores_by_file, good_by_file, bad_by_file = score_class(
        grouped_bank={1: [(feature_path.name, str(feature_path))]},
        target_worklist=[],
        model=None,
        device="cpu",
        query_chunk=8,
        sim_floor=0.0,
        target_batch_size=1,
        num_workers=0,
        keep_threshold=None,
        min_matches=3,
        target_references=None,
        early_accept=False,
        selection_mode="top-k-images",
        class_name="crop",
    )

    assert torch.equal(scores_by_file[feature_path.name], torch.ones(3))
    assert torch.equal(good_by_file[feature_path.name], torch.full((3,), 3, dtype=torch.int32))
    assert torch.equal(bad_by_file[feature_path.name], torch.zeros(3, dtype=torch.int32))
    assert torch.equal(
        _single_reference_keep_mask(num_features=3, top_k_features=2),
        torch.tensor([True, True, False]),
    )


def test_generic_layout_isolates_shot_artifacts(tmp_path):
    layout = DatasetLayout(
        dataset_root=str(tmp_path / "dataset"),
        train_ann_file=str(tmp_path / "dataset" / "train.json"),
        val_ann_file=str(tmp_path / "dataset" / "test.json"),
        train_image_dir=str(tmp_path / "dataset"),
        val_image_dir=str(tmp_path / "dataset"),
        artifact_root=str(tmp_path / "dataset" / "segrag_runs" / "strict_30shot"),
    )

    assert "strict_30shot" in layout.raw_feature_bank_dir
    assert "strict_30shot" in layout.scored_feature_bank_dir
    assert "strict_30shot" in layout.filtered_feature_bank_dir
