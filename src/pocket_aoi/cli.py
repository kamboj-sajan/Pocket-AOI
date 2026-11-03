"""Command line entry point. Block 0 has the two smoke tests; later blocks add setup-board, inspect and eval."""

import argparse
import io
import random
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from pocket_aoi.config import REPO_ROOT, load_config
from pocket_aoi.io import load_rgb

# Downscale real photos before sending: a 12 MP phone JPEG is far larger than the model can use.
SMOKE_IMAGE_LONG_SIDE = 1536


class SmokeReply(BaseModel):
    text_seen: str
    description: str


def _synthetic_board(label: str) -> bytes:
    """Draw a green board with one silkscreen label, so reading it back proves the image arrived."""
    # Imported here, not at the top of the file: a Python import runs code at runtime, so
    # commands that do not draw skip the cost.
    import cv2
    import numpy as np

    canvas = np.full((480, 640, 3), (60, 110, 30), dtype=np.uint8)
    cv2.rectangle(canvas, (220, 220), (420, 280), (150, 200, 230), thickness=-1)
    cv2.putText(canvas, label, (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4, cv2.LINE_AA)
    _, png = cv2.imencode(".png", canvas)
    return png.tobytes()


def _photo_as_jpeg(path: Path) -> bytes:
    """Load a photo upright, shrink it and encode it as JPEG bytes."""
    photo = load_rgb(path)
    photo.thumbnail((SMOKE_IMAGE_LONG_SIDE, SMOKE_IMAGE_LONG_SIDE))
    buffer = io.BytesIO()
    photo.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _cmd_smoke_llm(args: argparse.Namespace) -> int:
    """Send one image to Muse Glimmer and print the reply."""
    from openai import OpenAIError

    from pocket_aoi.llm import LLMClient, LLMSettings, extract_json, load_prompt

    cfg = load_config()
    try:
        settings = LLMSettings.from_config(cfg)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    expected_label = None
    if args.image:
        image = _photo_as_jpeg(args.image)
        source = str(args.image)
    else:
        expected_label = f"R{random.randint(10, 99)}"  # Random, so the model cannot pass by guessing.
        image = _synthetic_board(expected_label)
        source = f"synthetic board labelled {expected_label}"

    print(f"Backend : {settings.base_url}")
    print(f"Model   : {settings.model}")
    print(f"Image   : {source} ({len(image) / 1024:.0f} KB)")

    prompt = load_prompt("smoke", cfg["llm"]["prompt_versions"]["smoke"])
    client = LLMClient(settings)
    started = time.perf_counter()
    try:
        reply = client.complete(prompt, image=image, schema=SmokeReply, use_cache=False)
    except OpenAIError as error:
        print(f"ERROR: the backend call failed: {error}", file=sys.stderr)
        return 1
    print(f"Latency : {time.perf_counter() - started:.1f} s")
    print(f"Schema  : {'accepted' if client.use_response_format else 'rejected; relying on the prompt and pydantic'}")
    print(f"Reply   : {reply}")

    parsed = SmokeReply.model_validate_json(extract_json(reply))
    print(f"Parsed  : text_seen={parsed.text_seen!r} description={parsed.description!r}")

    if expected_label and expected_label not in parsed.text_seen:
        print(f"FAIL: expected the model to read {expected_label}.", file=sys.stderr)
        return 1
    print("PASS" if expected_label is None else f"PASS: the model read {expected_label}, so the image reached it.")
    return 0


def _cmd_smoke_sam(args: argparse.Namespace) -> int:
    """Load SAM 3.1, segment one image with one text prompt, and print the shapes."""
    from pocket_aoi.segment import Segmenter, draw_overlay, resolve_model_id

    cfg = load_config()
    model_id = args.model or resolve_model_id(cfg)
    threshold = cfg["segment"]["score_threshold"] if args.threshold is None else args.threshold

    image = load_rgb(args.image)
    print(f"Image   : {args.image} ({image.width} x {image.height})")

    started = time.perf_counter()
    segmenter = Segmenter(model_id, device=args.device)
    print(f"Model   : {model_id} on {segmenter.device} as {segmenter.dtype}")
    print(f"Load    : {time.perf_counter() - started:.1f} s")

    started = time.perf_counter()
    detections = segmenter.segment(image, args.prompt, threshold, cfg["segment"]["mask_threshold"])
    print(f"Infer   : {time.perf_counter() - started:.1f} s for prompt {args.prompt!r} at threshold {threshold}")

    print(f"Found   : {len(detections)} instance(s)")
    if detections:
        print(f"Mask    : shape {detections[0].mask.shape}, type {detections[0].mask.dtype}")
    # sorted with key= is like std::sort with a comparator lambda; [:5] slices off the first five.
    for detection in sorted(detections, key=lambda d: d.score, reverse=True)[:5]:
        box = ", ".join(f"{value:.0f}" for value in detection.box)
        print(f"  score {detection.score:.2f}  box ({box})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw_overlay(image, detections).save(args.out)
    print(f"Overlay : {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pocket-aoi", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smoke_llm = commands.add_parser("smoke-llm", help="send one image to Muse Glimmer and print the reply")
    smoke_llm.add_argument("--image", type=Path, help="a photo to send; default is a generated test board")
    smoke_llm.set_defaults(handler=_cmd_smoke_llm)

    smoke_sam = commands.add_parser("smoke-sam", help="segment one image with one text prompt")
    smoke_sam.add_argument("--image", type=Path, required=True)
    smoke_sam.add_argument("--prompt", default="printed circuit board")
    smoke_sam.add_argument("--threshold", type=float, help="default is segment.score_threshold in the config")
    smoke_sam.add_argument("--model", help="default is SAM_MODEL_ID, then models.sam in the config")
    smoke_sam.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default picks the fastest available")
    smoke_sam.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "smoke" / "sam_overlay.png")
    smoke_sam.set_defaults(handler=_cmd_smoke_sam)

    args = parser.parse_args(argv)
    # Functions are ordinary values in Python, so set_defaults stored the right one for this command.
    return args.handler(args)
