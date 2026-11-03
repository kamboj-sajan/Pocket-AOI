"""SAM 3.1 wrapper: text prompts in; masks, boxes and scores out, tiled and merged (section 3, Stage 2).

All SAM calls in the project go through this module (AGENTS.md section 5, rule 3).
"""

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from transformers import Sam3Model, Sam3Processor

_POST_PROCESS_CHUNK = 16

View = tuple[int, int, int, int]  # x0, y0, x1, y1 in full-image pixels


@dataclass(frozen=True)
class Detection:
    """One instance SAM found, in the pixel coordinates of the image it was given."""

    mask: np.ndarray                         # H x W bool
    box: tuple[float, float, float, float]   # x0, y0, x1, y1
    score: float


@dataclass(frozen=True)
class EncodedImage:
    """An image after SAM's vision encoder, ready for any number of text prompts."""

    vision_embeds: Any
    original_sizes: list[list[int]]


@dataclass(frozen=True)
class PartMask:
    """One detected part in full-image coordinates, with its mask compressed as COCO RLE."""

    rle: dict[str, Any]
    box: tuple[float, float, float, float]
    score: float
    cls: str
    prompt: str


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
        self.seconds = {"encode": 0.0, "decode": 0.0}

    def encode(self, image: Image.Image) -> EncodedImage:
        """Run the slow vision encoder once, so each text prompt only pays for the decoder."""
        started = time.perf_counter()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.dtype)
        # A with-block is Python's RAII: gradient tracking stays off until the block ends.
        with torch.inference_mode():
            vision_embeds = self.model.get_vision_features(pixel_values=inputs["pixel_values"])
        self.seconds["encode"] += time.perf_counter() - started
        return EncodedImage(vision_embeds, inputs["original_sizes"].tolist())

    def segment_encoded(
        self,
        encoded: EncodedImage,
        text: str,
        threshold: float = 0.4,
        mask_threshold: float = 0.5,
    ) -> list[Detection]:
        """Return every instance of the text prompt that scores at or above the threshold."""
        started = time.perf_counter()
        text_inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(
                vision_embeds=encoded.vision_embeds,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
        detections = post_process_in_chunks(
            self.processor, outputs, threshold, mask_threshold, encoded.original_sizes
        )
        self.seconds["decode"] += time.perf_counter() - started
        return detections

    def segment(
        self,
        image: Image.Image,
        text: str,
        threshold: float = 0.4,
        mask_threshold: float = 0.5,
    ) -> list[Detection]:
        """Segment one image with one text prompt."""
        return self.segment_encoded(self.encode(image), text, threshold, mask_threshold)

    def find_board(
        self,
        image: Image.Image,
        prompt: str,
        threshold: float,
        mask_threshold: float,
        input_long_side: int,
    ) -> np.ndarray | None:
        """Return the largest board mask at the image's full size, or None if SAM finds no board."""
        # SAM resizes its input anyway, so shrinking first saves upsampling many masks to 25 MP.
        small = image.copy()
        small.thumbnail((input_long_side, input_long_side))
        detections = self.segment(small, prompt, threshold, mask_threshold)
        if not detections:
            return None
        largest = max(detections, key=lambda detection: int(detection.mask.sum()))
        full = Image.fromarray(largest.mask).resize(image.size, Image.Resampling.NEAREST)
        return np.asarray(full, dtype=bool)

    def segment_parts(
        self,
        image: Image.Image,
        class_prompts: Mapping[str, str],
        *,  # Everything after this bare * must be passed by name.
        threshold: float,
        mask_threshold: float,
        tile_grid: int,
        tile_overlap: float,
    ) -> list[PartMask]:
        """Run every prompt on the full image and on overlapping tiles, without merging yet."""
        width, height = image.size
        views: list[View] = [(0, 0, width, height), *tile_boxes(width, height, tile_grid, tile_overlap)]
        parts: list[PartMask] = []
        for view in views:
            encoded = self.encode(image.crop(view))
            # dict.items() yields (key, value) pairs, unpacked here like C++ structured bindings.
            for prompt, cls in class_prompts.items():
                for detection in self.segment_encoded(encoded, prompt, threshold, mask_threshold):
                    if not touches_inner_edge(detection.mask, view, width, height):
                        parts.append(_to_part_mask(detection, view, width, height, cls, prompt))
        return parts


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


def tile_boxes(width: int, height: int, grid: int, overlap: float) -> list[View]:
    """Split an image into grid x grid tiles, each sharing `overlap` of its size with its neighbour."""
    tile_width = width / (grid - (grid - 1) * overlap)
    tile_height = height / (grid - (grid - 1) * overlap)
    step_x, step_y = tile_width * (1 - overlap), tile_height * (1 - overlap)
    return [
        (
            round(column * step_x),
            round(row * step_y),
            round(column * step_x + tile_width),
            round(row * step_y + tile_height),
        )
        for row in range(grid)
        for column in range(grid)
    ]


def touches_inner_edge(mask: np.ndarray, view: View, width: int, height: int) -> bool:
    """Return True if a tile's mask runs into a tile edge inside the image, so the part is cut off.

    A cut-off part overlaps its whole version too little to merge as a duplicate. The tile
    overlap or the full-image view still sees that part whole.
    """
    x0, y0, x1, y1 = view
    return bool(
        (x0 > 0 and mask[:, 0].any())
        or (y0 > 0 and mask[0, :].any())
        or (x1 < width and mask[:, -1].any())
        or (y1 < height and mask[-1, :].any())
    )


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Compress an H x W bool mask to COCO RLE, with counts as text so it can go in JSON."""
    # COCO RLE runs down columns, so pycocotools needs Fortran (column-first) memory order.
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def rle_to_mask(rle: Mapping[str, Any]) -> np.ndarray:
    """Expand COCO RLE back to an H x W bool mask."""
    return mask_utils.decode(dict(rle)).astype(bool)


def rle_ious(rle: dict[str, Any], others: Sequence[dict[str, Any]]) -> np.ndarray:
    """Return the IoU of one mask against each of the others."""
    if not others:
        return np.zeros(0)
    return np.asarray(mask_utils.iou([rle], list(others), [0] * len(others))).ravel()


def _to_part_mask(
    detection: Detection, view: View, width: int, height: int, cls: str, prompt: str
) -> PartMask:
    """Move a detection from tile coordinates into full-image coordinates."""
    x0, y0, x1, y1 = view
    full = np.zeros((height, width), dtype=bool)
    full[y0:y1, x0:x1] = detection.mask
    bx0, by0, bx1, by1 = detection.box
    return PartMask(mask_to_rle(full), (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0), detection.score, cls, prompt)


def keep_plausible(
    parts: Sequence[PartMask],
    board_rle: dict[str, Any] | None,
    min_area_ratio: float,
    max_area_ratio: float,
    min_inside_board: float,
) -> list[PartMask]:
    """Drop masks that are too small, too large or mostly off the board."""
    if not parts:
        return []
    height, width = parts[0].rle["size"]
    board_area = float(mask_utils.area(board_rle)) if board_rle else float(width * height)

    kept = []
    for part in parts:
        area = float(mask_utils.area(part.rle))
        if not min_area_ratio <= area / board_area <= max_area_ratio:
            continue
        if board_rle:
            inside = float(mask_utils.area(mask_utils.merge([part.rle, board_rle], intersect=True)))
            if inside < min_inside_board * area:
                continue
        kept.append(part)
    return kept


def merge_duplicates(parts: Sequence[PartMask], iou_threshold: float) -> list[PartMask]:
    """Keep the highest-scoring mask of each overlapping group within one class."""
    kept: list[PartMask] = []
    for part in sorted(parts, key=lambda p: p.score, reverse=True):
        rivals = [other.rle for other in kept if other.cls == part.cls]
        if not rivals or rle_ious(part.rle, rivals).max() <= iou_threshold:
            kept.append(part)
    return kept


def resolve_class_conflicts(
    parts: Sequence[PartMask],
    iou_threshold: float,
    score_delta: float,
    specificity: Sequence[str],
) -> list[PartMask]:
    """When two prompts hit one object, keep the higher score, or the more specific class if close."""
    # A dict comprehension builds a lookup table; enumerate yields (index, item) pairs.
    rank = {cls: index for index, cls in enumerate(specificity)}
    kept: list[PartMask] = []
    for part in sorted(parts, key=lambda p: p.score, reverse=True):
        overlapping = np.flatnonzero(rle_ious(part.rle, [other.rle for other in kept]) > iou_threshold)
        if overlapping.size == 0:
            kept.append(part)
            continue
        rival_index = int(overlapping[0])
        rival = kept[rival_index]
        unranked = len(specificity)
        closer_and_specific = (
            rival.score - part.score <= score_delta
            and rank.get(part.cls, unranked) < rank.get(rival.cls, unranked)
        )
        if closer_and_specific:
            kept[rival_index] = part
    return kept


def select_parts(
    parts: Sequence[PartMask],
    segment_cfg: Mapping[str, Any],
    board_rle: dict[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> list[PartMask]:
    """Apply section 3, Stage 2, steps 2 to 4: per-class threshold, plausibility, then both merges."""
    per_class = segment_cfg["class_score_thresholds"] if thresholds is None else thresholds
    default = segment_cfg["score_threshold"]
    confident = [part for part in parts if part.score >= per_class.get(part.cls, default)]
    plausible = keep_plausible(
        confident,
        board_rle,
        segment_cfg["min_part_area_ratio"],
        segment_cfg["max_part_area_ratio"],
        segment_cfg["min_inside_board"],
    )
    merged = merge_duplicates(plausible, segment_cfg["merge_iou"])
    return resolve_class_conflicts(
        merged,
        segment_cfg["merge_iou"],
        segment_cfg["specific_class_score_delta"],
        segment_cfg["class_specificity"],
    )


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


CLASS_COLOURS = {
    "electrolytic_capacitor": (230, 40, 200),
    "ceramic_capacitor": (255, 170, 0),
    "resistor": (0, 220, 255),
    "diode": (255, 255, 0),
    "led": (255, 60, 60),
    "ic": (70, 255, 90),
    "transistor": (170, 110, 255),
    "other": (220, 220, 220),
}


def draw_parts(image: Image.Image, parts: Sequence[PartMask]) -> Image.Image:
    """Tint each part by class and label it with its class and score."""
    canvas = np.array(image.convert("RGB"), dtype=np.float32)
    for part in parts:
        mask = rle_to_mask(part.rle)
        canvas[mask] = 0.55 * canvas[mask] + 0.45 * np.array(CLASS_COLOURS.get(part.cls, (220, 220, 220)))

    overlay = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=20)
    for part in parts:
        colour = CLASS_COLOURS.get(part.cls, (220, 220, 220))
        draw.rectangle(part.box, outline=colour, width=2)
        label = f"{part.cls[:5]} {part.score:.2f}"
        draw.text((part.box[0], max(0, part.box[1] - 22)), label, fill=colour, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return overlay
