"""Check the chunked post-processing against the library's own result, with no model download."""

from types import SimpleNamespace

import numpy as np
import torch
from transformers import Sam3ImageProcessor

from pocket_aoi.segment import post_process_in_chunks

NUM_QUERIES = 40
TARGET_SIZES = [[60, 80]]


def _fake_outputs() -> SimpleNamespace:
    """Build model outputs with a mix of confident and rejected query slots."""
    generator = torch.Generator().manual_seed(0)
    return SimpleNamespace(
        pred_logits=torch.randn(1, NUM_QUERIES, generator=generator) * 3,
        pred_boxes=torch.rand(1, NUM_QUERIES, 4, generator=generator),
        pred_masks=torch.randn(1, NUM_QUERIES, 12, 16, generator=generator),
        presence_logits=torch.tensor([[4.0]]),
    )


def test_chunked_result_matches_the_library_called_on_everything() -> None:
    processor = Sam3ImageProcessor()
    outputs = _fake_outputs()

    expected = processor.post_process_instance_segmentation(
        outputs, threshold=0.4, mask_threshold=0.5, target_sizes=TARGET_SIZES
    )[0]
    detections = post_process_in_chunks(processor, outputs, 0.4, 0.5, TARGET_SIZES)

    assert 0 < len(detections) < NUM_QUERIES  # The fake data must exercise both outcomes.
    assert len(detections) == len(expected["scores"])
    for detection, mask, box, score in zip(
        detections, expected["masks"], expected["boxes"], expected["scores"]
    ):
        assert detection.mask.dtype == bool
        assert detection.mask.shape == (60, 80)
        assert np.array_equal(detection.mask, mask.numpy().astype(bool))
        assert np.allclose(detection.box, box.numpy())
        assert np.isclose(detection.score, score.item())


def test_no_instances_above_the_threshold_gives_an_empty_list() -> None:
    detections = post_process_in_chunks(Sam3ImageProcessor(), _fake_outputs(), 1.1, 0.5, TARGET_SIZES)
    assert detections == []
