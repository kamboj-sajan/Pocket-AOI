"""Runs the stages in order, timing each one, saving debug images and caching the slow model steps.

Stage 0 is setup_board. prepare_photo runs Stages 1 and 2, find_suspects Stage 3, and inspect_photo
runs everything from a photo to a finished report.
"""

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from pocket_aoi.align import (
    AlignResult,
    GoldenFrame,
    align_to_golden,
    canonicalise_golden,
    checkerboard_blend,
    filled_hull,
)
from pocket_aoi.compare import Finding, compare_boards, mask_centroid
from pocket_aoi.composite import build_composite, encode_png
from pocket_aoi.config import REPO_ROOT, config_hash
from pocket_aoi.consequence import add_consequences
from pocket_aoi.io import load_rgb
from pocket_aoi.llm import LLMClient
from pocket_aoi.netlist import find_component, load_netlist
from pocket_aoi.report import is_safe_to_power, save_report, sort_faults
from pocket_aoi.schemas import GoldenProfile, Netlist, Part, Report, Suspect
from pocket_aoi.segment import (
    PartMask,
    Segmenter,
    draw_parts,
    mask_to_rle,
    rle_to_mask,
    select_parts,
)
from pocket_aoi.verdict import Judged, judge_suspects

BOARDS_DIR = REPO_ROOT / "boards"
CACHE_DIR = REPO_ROOT / "runs" / "cache"


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


def locate_parts(segmenter: Segmenter, image: Image.Image, cfg: Mapping[str, Any]) -> list[PartMask]:
    """Run the locating prompt on the full image and the tiles, without selecting yet.

    Low-scoring masks are kept in the cache, so the threshold can be retuned without running SAM.
    """
    segment_cfg = cfg["segment"]
    return segmenter.segment_parts(
        image,
        {segment_cfg["locate_prompt"]: "other"},
        threshold=segment_cfg["locate_floor"],
        mask_threshold=segment_cfg["mask_threshold"],
        tile_grid=segment_cfg["tile_grid"],
        tile_overlap=segment_cfg["tile_overlap"],
    )


def select_located(
    parts: Sequence[PartMask], board_rle: dict[str, Any], cfg: Mapping[str, Any]
) -> list[PartMask]:
    """Keep plausible masks above the locating threshold, and merge duplicates."""
    segment_cfg = cfg["segment"]
    return select_parts(parts, segment_cfg, board_rle, {"other": segment_cfg["locate_threshold"]})


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    overlap_x = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap_y = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    overlap = overlap_x * overlap_y
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return overlap / max(first_area + second_area - overlap, 1e-6)


def _make_part(
    ref: str | None,
    cls: str,
    located: PartMask | None,
    label_box: Sequence[int] | None,
    size: tuple[int, int],
    netlist: Netlist | None,
    cfg: Mapping[str, Any],
) -> Part:
    """Build a golden Part, taking class and package from the netlist when it knows this ref."""
    width, height = size
    if located is not None:
        rle, score = located.rle, located.score
        # SAM's boxes can overhang the image by a pixel or two, and a negative slice index would wrap around.
        box = (
            max(int(located.box[0]), 0),
            max(int(located.box[1]), 0),
            min(int(located.box[2]), width),
            min(int(located.box[3]), height),
        )
    else:
        # SAM found nothing here, so the hand-drawn box stands in as the part's mask.
        x0, y0, x1, y1 = (int(value) for value in label_box)  # type: ignore[union-attr]
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        rle, score, box = mask_to_rle(mask), 0.0, (x0, y0, x1, y1)

    component = find_component(netlist, ref) if netlist and ref else None
    if component is not None:
        cls = component.type
    package = component.package if component is not None else None
    return Part(
        ref=ref,
        cls=cls,
        bbox=box,
        centroid=mask_centroid(rle),
        mask_rle=rle,
        score=score,
        package=package,
        polarized=cls in cfg["polarized_classes"],
    )


def build_golden_parts(
    located: Sequence[PartMask],
    labels: Sequence[Mapping[str, Any]],
    size: tuple[int, int],
    netlist: Netlist | None,
    cfg: Mapping[str, Any],
) -> list[Part]:
    """Give each labelled ref its best mask. Without labels, every mask gets a numbered ref.

    With labels, the masks left over are kept without a ref: real objects outside the inspection
    list, such as solder terminals, that must not read as EXTRA later.
    """
    if not labels:
        ordered = sorted(located, key=lambda part: (part.box[1], part.box[0]))
        return [
            _make_part(f"P{number:02d}", "other", part, None, size, netlist, cfg)
            for number, part in enumerate(ordered, start=1)
        ]

    remaining = list(located)
    parts: list[Part] = []
    for label in labels:
        # max with a key finds the best item; default= covers the case of nothing left.
        best = max(remaining, key=lambda part: _box_iou(part.box, label["box"]), default=None)
        if best is not None and _box_iou(best.box, label["box"]) < cfg["segment"]["label_match_iou"]:
            best = None
        if best is not None:
            remaining.remove(best)
        part = _make_part(label["ref"], label.get("cls", "other"), best, label["box"], size, netlist, cfg)
        parts.append(part.model_copy(update={"marker_hint": label.get("marker_hint")}))
    parts.extend(_make_part(None, "other", part, None, size, netlist, cfg) for part in remaining)
    return parts


def read_yaml(path: Path) -> Any:
    """Read a YAML file, returning an empty dict when it does not exist."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_netlist_if_any(board_dir: Path) -> Netlist | None:
    path = board_dir / "netlist.json"
    return load_netlist(path) if path.exists() else None


@dataclass(frozen=True)
class GoldenBoard:
    """Everything an inspection needs to know about one board design."""

    board_id: str
    profile: GoldenProfile
    image: np.ndarray
    board_rle: dict[str, Any]
    frame: GoldenFrame
    netlist: Netlist | None
    config: dict[str, Any] = field(default_factory=dict)


def setup_board(
    board_id: str,
    golden_photo: Path,
    netlist_path: Path | None,
    make_segmenter: Callable[[], Segmenter],
    cfg: Mapping[str, Any],
    boards_dir: Path = BOARDS_DIR,
) -> GoldenProfile:
    """Stage 0: find the parts on the golden photo, assign refs, and save golden_profile.json."""
    board_dir = boards_dir / board_id
    board_dir.mkdir(parents=True, exist_ok=True)
    if netlist_path is not None and netlist_path.resolve() != (board_dir / "netlist.json").resolve():
        shutil.copyfile(netlist_path, board_dir / "netlist.json")
    netlist = load_netlist_if_any(board_dir)

    segmenter = make_segmenter()
    photo = load_rgb(golden_photo)
    golden = load_golden_frame(segmenter, photo, cfg)
    canonical = Image.fromarray(golden.image)
    board_rle = mask_to_rle(filled_hull(golden.mask))

    raw = locate_parts(segmenter, canonical, cfg)
    located = select_located(raw, board_rle, cfg)
    labels = read_yaml(board_dir / "refs.yaml").get("parts", []) if (board_dir / "refs.yaml").exists() else []

    width, height = golden.size
    parts = build_golden_parts(located, labels, (width, height), netlist, cfg)
    profile = GoldenProfile(board_id=board_id, canonical_size=(width, height), parts=parts)

    photo.save(board_dir / "golden.jpg", quality=95)
    canonical.save(board_dir / "golden_canonical.png")
    (board_dir / "golden_profile.json").write_text(profile.model_dump_json(), encoding="utf-8")
    (board_dir / "golden_frame.json").write_text(
        json.dumps({"homography": golden.homography.tolist(), "board_rle": board_rle}), encoding="utf-8"
    )
    overlay_parts = [
        PartMask(part.mask_rle, tuple(float(v) for v in part.bbox), part.score, part.cls, part.ref or "known object")
        for part in parts
    ]
    draw_parts(canonical, overlay_parts).save(board_dir / "golden_parts.png")
    return profile


def load_golden_board(board_id: str, cfg: Mapping[str, Any], boards_dir: Path = BOARDS_DIR) -> GoldenBoard:
    """Load what setup_board saved."""
    board_dir = boards_dir / board_id
    profile_path = board_dir / "golden_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"{profile_path} not found. Run pocket-aoi setup-board first.")
    profile = GoldenProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
    image = np.asarray(Image.open(board_dir / "golden_canonical.png").convert("RGB"))
    saved = json.loads((board_dir / "golden_frame.json").read_text(encoding="utf-8"))
    board_rle = saved["board_rle"]
    frame = GoldenFrame(
        image=image,
        mask=rle_to_mask(board_rle),
        homography=np.array(saved["homography"]),
        size=tuple(profile.canonical_size),
    )
    return GoldenBoard(board_id, profile, image, board_rle, frame, load_netlist_if_any(board_dir), dict(cfg))


def relabel_board(board_id: str, cfg: Mapping[str, Any], boards_dir: Path = BOARDS_DIR) -> GoldenProfile:
    """Rebuild golden_profile.json from refs.yaml, reusing the masks already found, so SAM need not run."""
    board_dir = boards_dir / board_id
    old = load_golden_board(board_id, cfg, boards_dir)
    labels = read_yaml(board_dir / "refs.yaml").get("parts", [])
    located = [PartMask(part.mask_rle, tuple(float(v) for v in part.bbox), part.score, part.cls, "") for part in old.profile.parts]
    width, height = old.profile.canonical_size
    parts = build_golden_parts(located, labels, (width, height), old.netlist, cfg)
    profile = GoldenProfile(board_id=board_id, canonical_size=(width, height), parts=parts)
    (board_dir / "golden_profile.json").write_text(profile.model_dump_json(), encoding="utf-8")
    overlay_parts = [
        PartMask(part.mask_rle, tuple(float(v) for v in part.bbox), part.score, part.cls, part.ref or "known object")
        for part in parts
    ]
    draw_parts(Image.fromarray(old.image), overlay_parts).save(board_dir / "golden_parts.png")
    return profile


@dataclass(frozen=True)
class PreparedPhoto:
    """A test photo after the slow model steps: aligned into the golden frame, with raw part masks."""

    ok: bool
    message: str
    image: np.ndarray | None
    homography: np.ndarray | None
    raw_parts: list[PartMask]
    board_iou: float
    tilt_ratio: float
    sift_inliers: int
    sift_error_px: float
    sift_accepted: bool
    seconds: dict[str, float]


def _prepare_cache_key(photo_path: Path, board_id: str, golden_image: np.ndarray, cfg: Mapping[str, Any]) -> str:
    """Key the cache on the photo, the golden image and only the settings the slow steps depend on."""
    digest = hashlib.sha256(photo_path.read_bytes())
    digest.update(board_id.encode("utf-8"))
    digest.update(golden_image[::16, ::16].tobytes())
    segment_cfg = cfg["segment"]
    relevant = {
        "align": cfg["align"],
        "models": cfg["models"],
        **{key: segment_cfg[key] for key in ("board_prompt", "locate_prompt", "locate_floor", "mask_threshold", "tile_grid", "tile_overlap")},
    }
    digest.update(json.dumps(relevant, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:24]


def prepare_photo(
    photo_path: Path,
    board: GoldenBoard,
    make_segmenter: Callable[[], Segmenter],
    cfg: Mapping[str, Any],
    use_cache: bool = True,
) -> PreparedPhoto:
    """Run Stages 1 and 2 on a test photo, reusing an earlier run of the same photo when possible."""
    key = _prepare_cache_key(photo_path, board.board_id, board.image, cfg)
    cache_dir = CACHE_DIR / board.board_id / key
    meta_path = cache_dir / "meta.json"
    if use_cache and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        image_path = cache_dir / "aligned.png"
        return PreparedPhoto(
            ok=meta["ok"],
            message=meta["message"],
            image=np.asarray(Image.open(image_path).convert("RGB")) if image_path.exists() else None,
            homography=np.array(meta["homography"]) if meta["homography"] is not None else None,
            raw_parts=[PartMask(p["rle"], tuple(p["box"]), p["score"], p["cls"], p["prompt"]) for p in meta["raw_parts"]],
            board_iou=meta["board_iou"],
            tilt_ratio=meta["tilt_ratio"],
            sift_inliers=meta["sift_inliers"],
            sift_error_px=meta["sift_error_px"],
            sift_accepted=meta["sift_accepted"],
            seconds=meta["seconds"],
        )

    segmenter = make_segmenter()
    started = time.perf_counter()
    result = align_photo(segmenter, load_rgb(photo_path), board.frame, cfg)
    align_seconds = time.perf_counter() - started

    started = time.perf_counter()
    raw_parts = locate_parts(segmenter, Image.fromarray(result.image), cfg) if result.ok and result.image is not None else []
    segment_seconds = time.perf_counter() - started

    refinement = result.refinement
    prepared = PreparedPhoto(
        ok=result.ok,
        message=result.message,
        image=result.image,
        homography=result.homography,
        raw_parts=raw_parts,
        board_iou=result.board_iou,
        tilt_ratio=result.tilt_ratio,
        sift_inliers=refinement.inliers if refinement else 0,
        sift_error_px=refinement.median_error_px if refinement else 0.0,
        sift_accepted=refinement.accepted if refinement else False,
        seconds={"align": align_seconds, "segment": segment_seconds},
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    if prepared.image is not None:
        Image.fromarray(prepared.image).save(cache_dir / "aligned.png")
    meta_path.write_text(
        json.dumps(
            {
                "ok": prepared.ok,
                "message": prepared.message,
                "homography": prepared.homography.tolist() if prepared.homography is not None else None,
                "raw_parts": [asdict(part) for part in prepared.raw_parts],
                "board_iou": prepared.board_iou,
                "tilt_ratio": prepared.tilt_ratio,
                "sift_inliers": prepared.sift_inliers,
                "sift_error_px": prepared.sift_error_px,
                "sift_accepted": prepared.sift_accepted,
                "seconds": prepared.seconds,
            }
        ),
        encoding="utf-8",
    )
    return prepared


def find_suspects(prepared: PreparedPhoto, board: GoldenBoard, cfg: Mapping[str, Any]) -> list[Finding]:
    """Stage 3 on an aligned photo. Fast, because the model steps are already done."""
    test_parts = select_located(prepared.raw_parts, board.board_rle, cfg)
    return compare_boards(board.profile, board.image, prepared.image, test_parts, cfg["compare"])


def save_suspects(
    findings: Sequence[Finding],
    prepared: PreparedPhoto,
    board: GoldenBoard,
    run_dir: Path,
    cfg: Mapping[str, Any],
) -> list[Suspect]:
    """Write one composite per finding and return the Suspect records that point to them."""
    composites_dir = run_dir / "composites"
    composites_dir.mkdir(parents=True, exist_ok=True)
    suspects = []
    for number, finding in enumerate(findings, start=1):
        composite = build_composite(
            board.image,
            prepared.image,
            finding.bbox,
            finding.part_bbox,
            cfg["verdict"]["composite_min_part_px"],
            cfg["verdict"]["composite_max_side_px"],
        )
        name = f"{number:02d}_{finding.golden_ref or 'extra'}_{finding.cv_finding}.png"
        path = composites_dir / name
        path.write_bytes(encode_png(composite))
        suspects.append(
            Suspect(
                golden_ref=finding.golden_ref,
                cv_finding=finding.cv_finding,
                bbox=finding.part_bbox,
                evidence=finding.evidence,
                composite_path=str(path),
            )
        )
    return suspects


def save_debug_images(prepared: PreparedPhoto, board: GoldenBoard, run_dir: Path, cfg: Mapping[str, Any]) -> None:
    """Save the aligned photo, a checkerboard blend and the located masks for inspection."""
    debug = run_dir / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    if prepared.image is None:
        return
    Image.fromarray(prepared.image).save(debug / "aligned.png")
    Image.fromarray(np.asarray(checkerboard_blend(board.image, prepared.image))).save(debug / "checkerboard.png")
    test_parts = select_located(prepared.raw_parts, board.board_rle, cfg)
    draw_parts(Image.fromarray(prepared.image), test_parts).save(debug / "test_masks.png")


def save_suspects_json(suspects: Sequence[Suspect], run_dir: Path) -> None:
    """Write suspects.json, so Stage 4 can be rerun alone."""
    # join builds one string from a generator; each suspect is already its own JSON text.
    body = ",\n".join(suspect.model_dump_json() for suspect in suspects)
    (run_dir / "suspects.json").write_text(f"[{body}]", encoding="utf-8")


def save_votes(judged: Sequence[Judged], run_dir: Path) -> None:
    """Write both votes per suspect, including those Glimmer dropped, for the ablation."""
    votes = [
        {
            "ref": item.suspect.golden_ref,
            "cv_finding": item.suspect.cv_finding,
            "verdict": item.verdict.model_dump(),
            "status": item.fault.status if item.fault else "dropped",
        }
        for item in judged
    ]
    (run_dir / "votes.json").write_text(json.dumps(votes, indent=1), encoding="utf-8")


def report_versions(client: LLMClient, segmenter_id: str, cfg: Mapping[str, Any]) -> dict[str, str]:
    """Record the model ids, prompt versions and config hash each report came from."""
    prompts = cfg["llm"]["prompt_versions"]
    return {
        "sam": segmenter_id,
        "llm": client.settings.model,
        "verdict_prompt": prompts["verdict"],
        "consequence_prompt": prompts["consequence"],
        "config": config_hash(),
    }


@dataclass(frozen=True)
class Inspection:
    """Everything one inspection produced. A rejected photo has a message and no report."""

    ok: bool
    message: str
    run_dir: Path
    report: Report | None = None
    judged: tuple[Judged, ...] = ()
    paths: Mapping[str, Path] = field(default_factory=dict)


def inspect_photo(
    photo_path: Path,
    board: GoldenBoard,
    make_segmenter: Callable[[], Segmenter],
    client: LLMClient,
    cfg: Mapping[str, Any],
    run_dir: Path,
    use_cache: bool = True,
) -> Inspection:
    """Run Stages 1 to 6 on one photo, and write the report into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_photo(photo_path, board, make_segmenter, cfg, use_cache=use_cache)
    save_debug_images(prepared, board, run_dir, cfg)
    if not prepared.ok or prepared.homography is None:
        return Inspection(False, prepared.message, run_dir)

    timings = dict(prepared.seconds)

    started = time.perf_counter()
    findings = find_suspects(prepared, board, cfg)
    suspects = save_suspects(findings, prepared, board, run_dir, cfg)
    save_suspects_json(suspects, run_dir)
    timings["compare"] = time.perf_counter() - started

    started = time.perf_counter()
    judged = judge_suspects(client, suspects, board.profile, board.netlist, cfg)
    save_votes(judged, run_dir)
    timings["verdicts"] = time.perf_counter() - started

    started = time.perf_counter()
    faults = add_consequences(client, [item.fault for item in judged if item.fault], board.netlist, cfg)
    timings["consequences"] = time.perf_counter() - started

    started = time.perf_counter()
    report = Report(
        board_id=board.board_id,
        test_image=str(photo_path),
        safe_to_power=is_safe_to_power(faults),
        faults=sort_faults(faults),
        timings_s=timings,
        versions=report_versions(client, cfg["models"]["sam"], cfg),
    )
    photo = np.asarray(load_rgb(photo_path))
    paths = save_report(report, photo, prepared.homography, run_dir, cfg)
    report.timings_s["report"] = time.perf_counter() - started
    # Written again, because the time spent writing the report is only known now.
    paths["report.json"].write_text(report.model_dump_json(indent=1), encoding="utf-8")
    return Inspection(True, prepared.message, run_dir, report, tuple(judged), paths)
