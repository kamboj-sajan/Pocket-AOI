"""Command line entry point: smoke tests, calibration, alignment, setup-board, inspect and eval."""

import argparse
import io
import random
import sys
import time
from collections.abc import Callable
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


def _cmd_spike(args: argparse.Namespace) -> int:
    """Calibrate Stage 2 on every golden photo under the kits folder."""
    from pocket_aoi.spike import run_spike

    cfg = load_config()
    run_spike(args.kits_dir, args.boards_dir, args.out, _segmenter_factory(cfg), cfg, fresh=args.fresh)
    return 0


def _cmd_align(args: argparse.Namespace) -> int:
    """Warp a photo into the golden frame, and save the result with a checkerboard blend."""
    import numpy as np
    from PIL import Image

    from pocket_aoi.align import checkerboard_blend
    from pocket_aoi.pipeline import align_photo, load_golden_frame
    from pocket_aoi.segment import Segmenter, resolve_model_id

    cfg = load_config()
    segmenter = Segmenter(resolve_model_id(cfg))
    started = time.perf_counter()
    golden = load_golden_frame(segmenter, load_rgb(args.golden), cfg)
    result = align_photo(segmenter, load_rgb(args.photo), golden, cfg)
    print(f"Canonical : {golden.size[0]} x {golden.size[1]}")

    refinement = result.refinement
    if refinement is None:
        print("SIFT      : not enough matches, kept the quad-only warp")
    else:
        verdict = "accepted" if refinement.accepted else "rejected, kept the quad-only warp"
        print(f"SIFT      : {refinement.inliers} inliers, median error {refinement.median_error_px:.2f} px, {verdict}")
    print(f"Board IoU : {result.board_iou:.3f}")
    print(f"Tilt      : opposite board edges differ by {result.tilt_ratio:.2f} times")
    print(f"Time      : {time.perf_counter() - started:.1f} s")
    print(f"Result    : {result.message}")

    out_dir = args.out or REPO_ROOT / "runs" / "align" / args.photo.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(golden.image).save(out_dir / "golden_canonical.png")
    if result.image is not None:
        Image.fromarray(result.image).save(out_dir / "aligned.png")
        Image.fromarray(np.asarray(checkerboard_blend(golden.image, result.image))).save(out_dir / "checkerboard.png")
        print(f"Saved     : aligned.png and checkerboard.png in {out_dir}")
    return 0 if result.ok else 1


def _segmenter_factory(cfg: dict) -> Callable[[], object]:
    """Return a function that loads SAM at most once, and only if something needs it."""
    from functools import cache

    from pocket_aoi.segment import Segmenter, resolve_model_id

    # @cache remembers the first result, so later calls return the same loaded model.
    @cache
    def make_segmenter() -> Segmenter:
        return Segmenter(resolve_model_id(cfg))

    return make_segmenter


def _cmd_setup_board(args: argparse.Namespace) -> int:
    """Stage 0: build golden_profile.json for one board design."""
    from pocket_aoi.pipeline import setup_board

    cfg = load_config()
    profile = setup_board(args.board_id, args.golden, args.netlist, _segmenter_factory(cfg), cfg)
    inspected = [part for part in profile.parts if part.ref is not None]
    print(f"Board     : {args.board_id}, canonical {profile.canonical_size[0]} x {profile.canonical_size[1]}")
    print(f"Parts     : {len(inspected)} to inspect, {len(profile.parts) - len(inspected)} other objects known")
    weak = [part.ref for part in inspected if part.score == 0.0]
    if weak:
        print(f"No mask   : {', '.join(weak)} use their hand-drawn box")
    print(f"Saved     : boards/{args.board_id}/golden_profile.json and golden_parts.png")
    return 0


def _cmd_relabel(args: argparse.Namespace) -> int:
    """Rebuild a golden profile after editing refs.yaml, without running SAM."""
    from pocket_aoi.pipeline import relabel_board

    profile = relabel_board(args.board_id, load_config())
    inspected = [part for part in profile.parts if part.ref is not None]
    print(f"Board     : {args.board_id}")
    print(f"Parts     : {len(inspected)} to inspect, {len(profile.parts) - len(inspected)} other objects known")
    weak = [part.ref for part in inspected if part.score == 0.0]
    if weak:
        print(f"No mask   : {', '.join(weak)} use their hand-drawn box")
    return 0


def _make_client(cfg: dict) -> object:
    """Build the Muse Glimmer client from .env and the config."""
    from pocket_aoi.llm import LLMClient, LLMSettings

    return LLMClient(LLMSettings.from_config(cfg))


def _cmd_inspect_cv(args: argparse.Namespace, cfg: dict) -> int:
    """Run Stages 1 to 3 on one photo, and write suspects.json with one composite per suspect."""
    from pocket_aoi.pipeline import (
        find_suspects,
        load_golden_board,
        prepare_photo,
        save_debug_images,
        save_suspects,
        save_suspects_json,
    )

    board = load_golden_board(args.board_id, cfg)
    make_segmenter = _segmenter_factory(cfg)
    started = time.perf_counter()
    prepared = prepare_photo(args.photo, board, make_segmenter, cfg, use_cache=not args.no_cache)
    prepare_seconds = time.perf_counter() - started
    run_dir = REPO_ROOT / "runs" / f"{args.board_id}_{args.photo.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_debug_images(prepared, board, run_dir, cfg)

    if not prepared.ok:
        print(f"Result    : {prepared.message}")
        return 1

    started = time.perf_counter()
    findings = find_suspects(prepared, board, cfg)
    suspects = save_suspects(findings, prepared, board, run_dir, cfg)
    compare_seconds = time.perf_counter() - started
    save_suspects_json(suspects, run_dir)

    print(f"Aligned   : board IoU {prepared.board_iou:.3f}, {prepared.sift_inliers} SIFT inliers, tilt {prepared.tilt_ratio:.2f}")
    print(f"Time      : align and segment {prepare_seconds:.1f} s, compare {compare_seconds:.1f} s")
    print(f"Suspects  : {len(suspects)}")
    for suspect in suspects:
        print(f"  {suspect.golden_ref or 'unknown part'}: {suspect.cv_finding}, SSIM {suspect.evidence['ssim']:.2f}")
    print(f"Saved     : {run_dir}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect one photo end to end, or stop after Stage 3 with --cv-only."""
    from openai import APIError

    from pocket_aoi.pipeline import inspect_photo, load_golden_board
    from pocket_aoi.report import banner_text, fault_label

    cfg = load_config()
    if args.cv_only:
        return _cmd_inspect_cv(args, cfg)

    try:
        client = _make_client(cfg)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    board = load_golden_board(args.board_id, cfg)
    run_dir = REPO_ROOT / "runs" / f"{args.board_id}_{args.photo.stem}"
    try:
        result = inspect_photo(args.photo, board, _segmenter_factory(cfg), client, cfg, run_dir, use_cache=not args.no_cache)
    except APIError as error:
        print(f"ERROR: the Muse Glimmer backend refused the call: {error}", file=sys.stderr)
        return 1
    if not result.ok or result.report is None:
        print(f"Result    : {result.message}")
        return 1

    report = result.report
    print(f"Banner    : {banner_text(report)[0]}")
    print(f"Suspects  : {len(result.judged)}, of which {sum(item.fault is None for item in result.judged)} rejected by Muse Glimmer")
    for fault in report.faults:
        severity = fault.consequence.severity if fault.consequence else "none"
        print(f"  {fault_label(fault)}: {fault.status}, {severity}")
    # A generator expression inside join, like a loop that appends to a string.
    print("Time      : " + ", ".join(f"{stage} {seconds:.1f} s" for stage, seconds in report.timings_s.items()))
    print(f"Report    : {result.paths['report.html']}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Score the pipeline against ground truth on real photos."""
    from openai import APIError

    # eval/ sits beside src/, so its folder's parent must be on the import path.
    sys.path.insert(0, str(REPO_ROOT))
    from eval.run_eval import load_ground_truth, run_eval

    cfg = load_config()
    try:
        client = None if args.cv_only else _make_client(cfg)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    records = load_ground_truth(args.gt, args.split, args.board)
    print(f"{len(records)} photos")
    try:
        run_eval(records, _segmenter_factory(cfg), client, cfg, args.tag)
    except APIError as error:
        print(f"ERROR: the Muse Glimmer backend refused the call, so no results were written: {error}", file=sys.stderr)
        return 1
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

    spike = commands.add_parser("spike", help="count parts on each golden photo and suggest thresholds")
    spike.add_argument("--kits-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    spike.add_argument("--boards-dir", type=Path, default=REPO_ROOT / "boards")
    spike.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "spike")
    spike.add_argument("--fresh", action="store_true", help="run SAM again instead of reusing saved detections")
    spike.set_defaults(handler=_cmd_spike)

    align = commands.add_parser("align", help="warp a photo into the golden frame")
    align.add_argument("--golden", type=Path, required=True)
    align.add_argument("--photo", type=Path, required=True)
    align.add_argument("--out", type=Path, help="default is runs/align/<photo name>/")
    align.set_defaults(handler=_cmd_align)

    setup = commands.add_parser("setup-board", help="build the golden profile for one board design")
    setup.add_argument("--board-id", required=True)
    setup.add_argument("--golden", type=Path, required=True)
    setup.add_argument("--netlist", type=Path, help="default is boards/<board id>/netlist.json if it exists")
    setup.set_defaults(handler=_cmd_setup_board)

    relabel = commands.add_parser("relabel", help="rebuild a golden profile from refs.yaml without running SAM")
    relabel.add_argument("--board-id", required=True)
    relabel.set_defaults(handler=_cmd_relabel)

    inspect = commands.add_parser("inspect", help="inspect one photo against a board's golden profile")
    inspect.add_argument("--board-id", required=True)
    inspect.add_argument("--photo", type=Path, required=True)
    inspect.add_argument("--cv-only", action="store_true", help="stop after Stage 3, without calling Muse Glimmer")
    inspect.add_argument("--no-cache", action="store_true", help="run SAM again instead of reusing an earlier run")
    inspect.set_defaults(handler=_cmd_inspect)

    evaluate = commands.add_parser("eval", help="score the pipeline against ground truth")
    evaluate.add_argument("--gt", type=Path, default=REPO_ROOT / "eval" / "ground_truth")
    evaluate.add_argument("--split", choices=["dev", "test"], help="default is every photo")
    evaluate.add_argument("--board", help="default is every board")
    evaluate.add_argument("--tag", default="cv", help="results go to runs/eval/<tag>/")
    evaluate.add_argument("--cv-only", action="store_true", help="stop after Stage 3, without calling Muse Glimmer")
    evaluate.set_defaults(handler=_cmd_eval)

    args = parser.parse_args(argv)
    # Functions are ordinary values in Python, so set_defaults stored the right one for this command.
    return args.handler(args)
