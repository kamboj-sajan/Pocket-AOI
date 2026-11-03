"""Stage 4 input: side-by-side GOLDEN and TEST crops for Muse Glimmer."""

from collections.abc import Sequence

import cv2
import numpy as np

_LABEL_SCALE = 1.1
_LABEL_THICKNESS = 2
_SEPARATOR_PX = 8


def _label(crop: np.ndarray, text: str) -> np.ndarray:
    """Draw a label on a dark strip inside the image's top-left corner."""
    labelled = crop.copy()
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, _LABEL_SCALE, _LABEL_THICKNESS)
    cv2.rectangle(labelled, (0, 0), (text_width + 16, text_height + baseline + 12), (0, 0, 0), thickness=-1)
    cv2.putText(
        labelled, text, (8, text_height + 6), cv2.FONT_HERSHEY_SIMPLEX, _LABEL_SCALE, (255, 255, 255), _LABEL_THICKNESS, cv2.LINE_AA
    )
    return labelled


def composite_scale(part_box: Sequence[int], crop_size: tuple[int, int], min_part_px: int, max_side_px: int) -> float:
    """Choose a scale so the part spans at least min_part_px, without the crop passing max_side_px."""
    part_side = max(part_box[2] - part_box[0], part_box[3] - part_box[1], 1)
    crop_side = max(crop_size)
    return max(min(min_part_px / part_side, max_side_px / max(crop_side, 1)), 1.0)


def build_composite(
    golden_image: np.ndarray,
    test_image: np.ndarray,
    crop_box: Sequence[int],
    part_box: Sequence[int],
    min_part_px: int,
    max_side_px: int,
) -> np.ndarray:
    """Return one RGB image: golden crop on the left, test crop on the right, at the same scale."""
    x0, y0, x1, y1 = crop_box
    golden_crop = golden_image[y0:y1, x0:x1]
    test_crop = test_image[y0:y1, x0:x1]
    height, width = golden_crop.shape[:2]
    scale = composite_scale(part_box, (width, height), min_part_px, max_side_px)
    size = (round(width * scale), round(height * scale))
    golden_large = _label(cv2.resize(golden_crop, size, interpolation=cv2.INTER_CUBIC), "GOLDEN")
    test_large = _label(cv2.resize(test_crop, size, interpolation=cv2.INTER_CUBIC), "TEST")
    separator = np.full((size[1], _SEPARATOR_PX, 3), 255, dtype=np.uint8)
    return np.hstack([golden_large, separator, test_large])


def encode_png(image: np.ndarray) -> bytes:
    """Encode an RGB array as PNG bytes."""
    # OpenCV expects BGR channel order, so the channels are reversed with a slice before encoding.
    ok, buffer = cv2.imencode(".png", image[:, :, ::-1])
    if not ok:
        raise ValueError("could not encode the composite as PNG")
    return buffer.tobytes()
