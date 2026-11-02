"""SAM 3.1 wrapper: one image and one text prompt in; masks, boxes and scores out.

All SAM calls in the project go through this module (AGENTS.md section 5, rule 3).
Tiling and merging (section 3, Stage 2) arrive in Block 1.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor

_POST_PROCESS_CHUNK = 16


@dataclass(frozen=True)
class Detection:
    """One instance SAM found, in the pixel coordinates of the image it was given."""

    mask: np.ndarray                         # H x W bool
    box: tuple[float, float, float, float]   # x0, y0, x1, y1
    score: float


def pick_device() -> tuple[str, torch.dtype]:
    """Choose the fastest available device and a float type it handles well."""
    if torch.cuda.is_available():
        return "cuda", torch.float16  # A Colab or Kaggle T4 has no native bfloat16.
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def resolve_model_id(cfg: Mapping[str, Any]) -> str:
    """Use SAM_MODEL_ID from .env when set, otherwise the id in the YAML config."""
    return os.environ.get("SAM_MODEL_ID") or cfg["models"]["sam"]


class Segmenter:
    """Holds a loaded SAM 3.1 model. Create one and reuse it, since loading takes a while."""

    def __init__(self, model_id: str, device: str | None = None, dtype: torch.dtype | None = None) -> None:
        auto_device, auto_dtype = pick_device()
        self.model_id = model_id
        self.device = device or auto_device
        self.dtype = dtype or (auto_dtype if self.device == auto_device else torch.float32)
        self.model = Sam3Model.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

    def segment(
        self,
        image: Image.Image,
        text: str,
        threshold: float = 0.4,
        mask_threshold: float = 0.5,
    ) -> list[Detection]:
        """Return every instance of the text prompt that scores at or above the threshold."""
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device, self.dtype)
        # A with-block is Python's RAII: gradient tracking stays off until the block ends.
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return post_process_in_chunks(
            self.processor, outputs, threshold, mask_threshold, inputs["original_sizes"].tolist()
        )


def post_process_in_chunks(
    processor: Any,
    outputs: Any,
    threshold: float,
    mask_threshold: float,
    target_sizes: list[list[int]],
) -> list[Detection]:
    """Run the library's post-processing on a few query slots at a time.

    The library upsamples every kept instance at once as 8-byte integers, so about 100 masks
    of a 1920 x 2925 photo ask for one 4.5 GiB buffer. Each chunk is copied to the CPU as
    1-byte bools before the next one starts, which keeps the peak small.
    """
    detections: list[Detection] = []
    for start in range(0, outputs.pred_logits.shape[1], _POST_PROCESS_CHUNK):
        # A slice object is a reusable start:stop, the same as writing [:, start:stop] on each array.
        chunk = slice(start, start + _POST_PROCESS_CHUNK)
        chunk_outputs = SimpleNamespace(
            pred_logits=outputs.pred_logits[:, chunk],
            pred_boxes=outputs.pred_boxes[:, chunk],
            pred_masks=outputs.pred_masks[:, chunk],
            presence_logits=outputs.presence_logits,
        )
        result = processor.post_process_instance_segmentation(
            chunk_outputs, threshold=threshold, mask_threshold=mask_threshold, target_sizes=target_sizes
        )[0]
        masks = result["masks"].cpu().numpy().astype(bool)
        boxes = result["boxes"].float().cpu().numpy()
        scores = result["scores"].float().cpu().numpy()
        # extend appends every item, like vector::insert at end(); zip walks the arrays in step.
        detections.extend(
            Detection(mask, tuple(box.tolist()), float(score))
            for mask, box, score in zip(masks, boxes, scores)
        )
    return detections


def draw_overlay(image: Image.Image, detections: Sequence[Detection]) -> Image.Image:
    """Tint each mask, then draw its box and score on top."""
    canvas = np.array(image.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(0)
    colours = [tuple(int(c) for c in rng.integers(60, 256, size=3)) for _ in detections]

    for detection, colour in zip(detections, colours):
        # Indexing with a bool array selects only the True pixels, with no explicit x, y loop.
        canvas[detection.mask] = 0.5 * canvas[detection.mask] + 0.5 * np.array(colour)

    overlay = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for detection, colour in zip(detections, colours):
        draw.rectangle(detection.box, outline=colour, width=3)
        draw.text((detection.box[0] + 4, detection.box[1] + 4), f"{detection.score:.2f}", fill=(255, 255, 255))
    return overlay
