"""Prompt B: read each golden part's reference designator from the silkscreen beside it.

A guess is never kept. A reply stands only if the ref is on the board and its prefix fits the class,
and every other part stays blank for a person to fill in (section 3, Stage 0, step 3).
"""

import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from pocket_aoi.composite import encode_png
from pocket_aoi.llm import LLMClient, load_prompt
from pocket_aoi.netlist import component_refs
from pocket_aoi.schemas import Netlist, Part

_CONTEXT_MARGIN = 1.0   # Silkscreen sits beside a part, so the crop shows a part-sized margin all round.
_MIN_LONG_SIDE_PX = 768

# Which reference prefixes each class may carry. "other" covers switches, connectors, pots and so on.
_PREFIXES: dict[str, tuple[str, ...]] = {
    "resistor": ("R",),
    "electrolytic_capacitor": ("C",),
    "ceramic_capacitor": ("C",),
    "diode": ("D",),
    "led": ("LED", "D"),
    "ic": ("U", "IC"),
    "transistor": ("Q", "T"),
}


class SilkscreenReply(BaseModel):
    ref: str | None
    alternatives: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


def accept_ref(ref: str | None, part_class: str, candidates: Sequence[str]) -> str | None:
    """Keep a reply only if the ref is on this board and its prefix fits the part's class."""
    if ref is None or ref not in candidates:
        return None
    # re.match anchors at the start; group() returns the letters it matched.
    match = re.match(r"[A-Za-z]+", ref)
    prefix = match.group().upper() if match else ""
    allowed = _PREFIXES.get(part_class)
    return ref if allowed is None or prefix in allowed else None


def outlined_crop(image: np.ndarray, part: Part) -> bytes:
    """Return the part outlined in red with the board around it, as PNG bytes."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = part.bbox
    margin_x, margin_y = int((x1 - x0) * _CONTEXT_MARGIN), int((y1 - y0) * _CONTEXT_MARGIN)
    left, top = max(x0 - margin_x, 0), max(y0 - margin_y, 0)
    right, bottom = min(x1 + margin_x, width), min(y1 + margin_y, height)

    crop = image[top:bottom, left:right].copy()
    cv2.rectangle(crop, (x0 - left, y0 - top), (x1 - left, y1 - top), (255, 0, 0), thickness=max(max(crop.shape[:2]) // 200, 2))
    scale = max(_MIN_LONG_SIDE_PX / max(crop.shape[:2]), 1.0)
    return encode_png(cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))


def read_refs(
    client: LLMClient,
    image: np.ndarray,
    parts: Sequence[Part],
    netlist: Netlist | None,
    cfg: Mapping[str, Any],
) -> list[str | None]:
    """Return one accepted ref or None per part, in the same order."""
    if netlist is None:
        return [None] * len(parts)
    prompt = load_prompt("silkscreen", cfg["llm"]["prompt_versions"]["silkscreen"])
    candidates = sorted(component_refs(netlist))

    def read(part: Part) -> str | None:
        values = {"part_class": part.cls, "candidate_refs": ", ".join(candidates)}
        reply = client.complete_json(SilkscreenReply, prompt, values, image=outlined_crop(image, part))
        return accept_ref(reply.ref, part.cls, candidates) if reply else None

    with ThreadPoolExecutor(max_workers=cfg["verdict"]["max_workers"]) as pool:
        refs = list(pool.map(read, parts))

    # Two parts cannot hold one ref, so a ref read twice is blanked on both for a person to settle.
    return [ref if ref is not None and refs.count(ref) == 1 else None for ref in refs]
