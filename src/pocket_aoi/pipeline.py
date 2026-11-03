"""Runs Stages 0 to 6 in order, timing each one and saving debug images. Stage 1 is in; the rest is Blocks 2 and 3."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

from pocket_aoi.align import AlignResult, GoldenFrame, align_to_golden, canonicalise_golden
from pocket_aoi.segment import Segmenter


def _board_mask(segmenter: Segmenter, photo: Image.Image, cfg: Mapping[str, Any]) -> np.ndarray | None:
    return segmenter.find_board(
        photo,
        cfg["segment"]["board_prompt"],
        cfg["align"]["board_threshold"],
        cfg["segment"]["mask_threshold"],
        cfg["align"]["board_input_long_side"],
    )


def load_golden_frame(segmenter: Segmenter, photo: Image.Image, cfg: Mapping[str, Any]) -> GoldenFrame:
    """Find the board in the golden photo and warp it into the canonical frame."""
    mask = _board_mask(segmenter, photo, cfg)
    if mask is None:
        raise ValueError("no board found in the golden photo")
    return canonicalise_golden(np.asarray(photo), mask, cfg["align"])


def align_photo(
    segmenter: Segmenter, photo: Image.Image, golden: GoldenFrame, cfg: Mapping[str, Any]
) -> AlignResult:
    """Find the board in a test photo and warp it into the golden frame."""
    return align_to_golden(np.asarray(photo), _board_mask(segmenter, photo, cfg), golden, cfg["align"])
