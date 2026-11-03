"""Run the pipeline on every ground-truth photo, and score it. Results land in runs/eval/<tag>/."""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from eval.metrics import (
    PhotoResult,
    box_covers_mask,
    format_ablation,
    format_failed_calls,
    format_misses,
    format_runtime,
    format_summary,
    same_type,
    summarise,
    summarise_by_label,
)
from pocket_aoi.align import warp_mask
from pocket_aoi.config import REPO_ROOT
from pocket_aoi.llm import LLMClient
from pocket_aoi.pipeline import (
    GoldenBoard,
    PreparedPhoto,
    find_suspects,
    load_golden_board,
    prepare_photo,
    save_suspects,
)
from pocket_aoi.schemas import GroundTruth
from pocket_aoi.segment import Segmenter
from pocket_aoi.verdict import NO_REPLY, judge_suspects


@dataclass(frozen=True)
class Claim:
    """One thing the pipeline says is wrong with a photo, from either pipeline in the ablation."""

    ref: str | None
    fault_type: str
    bbox: tuple[int, int, int, int]
    evidence: Mapping[str, float]


def load_ground_truth(folder: Path, split: str | None, board: str | None) -> list[GroundTruth]:
    """Read every ground-truth file under a folder, optionally for one split or board."""
    records = [
        GroundTruth.model_validate_json(path.read_text(encoding="utf-8")) for path in sorted(folder.rglob("*.json"))
    ]
    return [
        record
        for record in records
        if (split is None or record.split == split) and (board is None or record.board_id == board)
    ]


def _load_defect_mask(mask_path: str) -> np.ndarray:
    with Image.open(REPO_ROOT / mask_path) as image:
        return np.asarray(image.convert("L")) > 0


def share_in_frame(mask: np.ndarray, homography: np.ndarray, size: tuple[int, int]) -> float:
    """Return the share of a defect's pixels that land inside the canonical frame.

    A bent header pin hangs off the board, and the frame is cropped to the board's outline, so the
    pipeline never sees such a defect. Those are counted apart, rather than as ordinary misses.
    """
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return 0.0
    # np.stack joins the arrays into N x 2, and the extra axis is the shape perspectiveTransform expects.
    points = np.stack([columns, rows], axis=1).astype(np.float32)[:, np.newaxis, :]
    mapped = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
    width, height = size
    inside = (mapped[:, 0] >= 0) & (mapped[:, 0] < width) & (mapped[:, 1] >= 0) & (mapped[:, 1] < height)
    return float(inside.mean())


def _on_defect(claim: Claim, ref: str | None, canonical_mask: np.ndarray | None, min_overlap: float) -> bool:
    """A claim is on a defect when it names the same ref, or for mask labels when the two overlap."""
    if ref is not None:
        return claim.ref == ref
    return canonical_mask is not None and box_covers_mask(claim.bbox, canonical_mask, min_overlap)


def score_photo(
    truth: GroundTruth,
    prepared: PreparedPhoto,
    board: GoldenBoard,
    claims: Sequence[Claim],
    review: Sequence[Claim],
    eval_cfg: Mapping[str, Any],
) -> PhotoResult:
    """Compare one photo's claims with its ground truth. Items needing review are not claims of a fault."""
    result = PhotoResult(
        photo=truth.photo,
        board_id=truth.board_id,
        label=truth.label,
        split=truth.split,
        aligned=prepared.ok,
        message=prepared.message,
        needs_review=len(review),
        seconds=dict(prepared.seconds),
    )
    if not prepared.ok:
        result.planted = len(truth.defects)
        result.missed = [defect.type for defect in truth.defects]
        return result

    min_overlap = eval_cfg["min_defect_overlap"]
    claimed: set[int] = set()
    for defect in truth.defects:
        canonical_mask = None
        off_board = False
        if defect.mask is not None:
            mask = _load_defect_mask(defect.mask)
            off_board = share_in_frame(mask, prepared.homography, board.frame.size) < eval_cfg["min_in_frame_share"]
            canonical_mask = warp_mask(mask, prepared.homography, board.frame.size)

        # enumerate yields (index, item), and the comprehension keeps the indices of the matching claims.
        hits = [i for i, claim in enumerate(claims) if _on_defect(claim, defect.ref, canonical_mask, min_overlap)]
        claimed.update(hits)

        if off_board:
            result.off_board += 1
            result.off_board_found += bool(hits)
            continue

        result.planted += 1
        if hits:
            result.found_any += 1
        else:
            result.missed.append(defect.type)
            result.found_in_review += any(_on_defect(item, defect.ref, canonical_mask, min_overlap) for item in review)
        if any(same_type(claims[i].fault_type, defect.type, defect.accept) for i in hits):
            result.found_right_type += 1

    result.extra_findings = len(claims) - len(claimed)
    result.findings = [
        {"ref": c.ref, "type": c.fault_type, "bbox": list(c.bbox), "evidence": dict(c.evidence), "on_defect": i in claimed}
        for i, c in enumerate(claims)
    ]
    return result


def _photo_name(truth: GroundTruth) -> str:
    """Name a photo by its board and file, since VisA reuses file names across boards."""
    return f"{truth.board_id}_{Path(truth.photo).stem}"


def run_eval(
    records: Sequence[GroundTruth],
    make_segmenter: Callable[[], Segmenter],
    client: LLMClient | None,
    cfg: Mapping[str, Any],
    tag: str,
    verbose: bool = True,
) -> dict[str, list[PhotoResult]]:
    """Score Stage 3 alone on every photo and, with a client, CV plus Muse Glimmer from the same run."""
    out_dir = REPO_ROOT / "runs" / "eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    boards: dict[str, GoldenBoard] = {}
    cv_results: list[PhotoResult] = []
    fused_results: list[PhotoResult] = []
    eval_cfg = cfg["eval"]

    for number, truth in enumerate(records, start=1):
        # setdefault would load the board every time, because its argument is computed first.
        if truth.board_id not in boards:
            boards[truth.board_id] = load_golden_board(truth.board_id, cfg)
        board = boards[truth.board_id]
        prepared = prepare_photo(REPO_ROOT / truth.photo, board, make_segmenter, cfg)

        findings = find_suspects(prepared, board, cfg) if prepared.ok else []
        cv_claims = [Claim(f.golden_ref, f.cv_finding, f.part_bbox, f.evidence) for f in findings]
        cv_result = score_photo(truth, prepared, board, cv_claims, [], eval_cfg)
        cv_results.append(cv_result)
        line = f"cv {cv_result.found_any}/{cv_result.planted} extra {cv_result.extra_findings}"

        if client is not None:
            started = time.perf_counter()
            suspects = save_suspects(findings, prepared, board, out_dir / "photos" / _photo_name(truth), cfg)
            judged = judge_suspects(client, suspects, board.profile, board.netlist, cfg)
            confirmed = [i.fault for i in judged if i.fault and i.fault.status == "confirmed"]
            review = [i.fault for i in judged if i.fault and i.fault.status == "needs_review"]
            fused_result = score_photo(
                truth,
                prepared,
                board,
                [Claim(f.ref, f.fault_type, f.suspect.bbox, f.suspect.evidence) for f in confirmed],
                [Claim(f.ref, f.fault_type, f.suspect.bbox, f.suspect.evidence) for f in review],
                eval_cfg,
            )
            fused_result.seconds["verdicts"] = time.perf_counter() - started
            fused_result.failed_calls = sum(item.verdict.evidence == NO_REPLY for item in judged)
            fused_results.append(fused_result)
            line += f" | fused {fused_result.found_any}/{fused_result.planted} extra {fused_result.extra_findings} review {fused_result.needs_review}"

        if verbose:
            state = "" if prepared.ok else f" REJECTED: {prepared.message}"
            print(f"[{number}/{len(records)}] {_photo_name(truth)} {truth.label}: {line}{state}", flush=True)

    final = fused_results if client is not None else cv_results
    (out_dir / "results_cv.json").write_text(json.dumps([asdict(r) for r in cv_results], indent=1), encoding="utf-8")
    sections = ["# CV only", "", format_summary(summarise(cv_results)), "", format_summary(summarise_by_label(cv_results), "Defect")]
    if client is not None:
        (out_dir / "results_fused.json").write_text(json.dumps([asdict(r) for r in fused_results], indent=1), encoding="utf-8")
        sections += [
            "",
            "# CV + Muse Glimmer",
            "",
            format_summary(summarise(fused_results)),
            "",
            format_summary(summarise_by_label(fused_results), "Defect"),
            "",
            "# Ablation",
            "",
            format_ablation(cv_results, fused_results),
            "",
            format_failed_calls(fused_results),
        ]
    sections += ["", "# Runtime", "", format_runtime(final)]
    summary_text = "\n".join(sections)
    (out_dir / "summary.md").write_text(summary_text + "\n", encoding="utf-8")
    (out_dir / "misses.md").write_text(format_misses(final) + "\n", encoding="utf-8")
    print("\n" + summary_text)
    return {"cv": cv_results, "fused": fused_results}
