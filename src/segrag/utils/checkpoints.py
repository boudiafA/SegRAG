"""Resolve the tested SAM 3 weights without following a moving Hub revision."""

from __future__ import annotations

import os
from pathlib import Path

from segrag.reproducibility.protocol import sha256_file


SAM3_MODEL_ID = "facebook/sam3"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM3_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"


def resolve_sam3_checkpoint() -> str:
    checkpoint = os.environ.get("SEGRAG_SAM3_CHECKPOINT")
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(
            repo_id=SAM3_MODEL_ID, filename="sam3.pt", revision=SAM3_REVISION,
        ))
    if not path.is_file():
        raise FileNotFoundError(f"SAM 3 checkpoint not found: {path}")
    if sha256_file(path) != SAM3_SHA256:
        raise ValueError(
            f"SAM 3 checkpoint hash mismatch: {path}. Use the tested facebook/sam3 "
            f"revision {SAM3_REVISION}; refusing to silently use different weights."
        )
    return str(path)
