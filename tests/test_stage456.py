"""Stages 4 to 6 and the evaluation scorer, against a fake backend, so no network or GPU is needed."""

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from eval.make_result_sheets import make_sheet, make_tile
from eval.metrics import PhotoResult, format_ablation, summarise_by_label
from eval.run_eval import Claim, sample_clean_photos, score_photo, share_in_frame
from eval.run_polarity_eval import label_box, pick_shots, status_in_shot
from pocket_aoi.config import load_config
from pocket_aoi.consequence import (
    add_consequences,
    apply_floor,
    check_card,
    severity_floor,
    template_card,
    write_card,
)
from pocket_aoi.llm import LLMClient, LLMSettings
from pocket_aoi.pipeline import GoldenBoard, PreparedPhoto
from pocket_aoi.report import (
    banner_text,
    draw_report_overlay,
    fault_colour,
    is_safe_to_power,
    save_report,
    sort_faults,
)
from pocket_aoi.schemas import (
    Component,
    Consequence,
    Fault,
    GoldenProfile,
    GroundTruth,
    GroundTruthDefect,
    Net,
    Netlist,
    Part,
    Report,
    Suspect,
    Supply,
    Verdict,
)
from pocket_aoi.segment import mask_to_rle
from pocket_aoi.verdict import fuse, judge_suspects, prompt_values, unsure_verdict

CFG = load_config()
VERDICT_CFG = CFG["verdict"]

NETLIST = Netlist(
    board_id="flasher",
    description="555 astable LED flasher",
    supply=Supply(net="VCC", volts=9),
    components=[
        Component(ref="U1", value="NE555", type="ic", package="DIP-8", pins={"1": "GND", "8": "VCC"}),
        Component(ref="C1", value="100uF", type="electrolytic_capacitor", package="radial", pins={"1": "+", "2": "-"}),
        Component(ref="R1", value="1K", type="resistor", package="axial", pins={"1": "1", "2": "2"}),
    ],
    nets=[
        Net(name="VCC", nodes=["U1.8", "C1.1", "R1.1"]),
        Net(name="GND", nodes=["U1.1", "C1.2"]),
    ],
)


def _suspect(ref: str | None = "C1", finding: str = "MISSING", composite: str = "") -> Suspect:
    return Suspect(
        golden_ref=ref,
        cv_finding=finding,
        bbox=(10, 10, 60, 60),
        evidence={"ssim": 0.31, "best_rotation_deg": 180.0, "offset_ratio": 0.0},
        composite_path=composite,
    )


def _verdict(kind: str, confidence: float = 0.9) -> Verdict:
    return Verdict(golden_observation="g", test_observation="t", verdict=kind, confidence=confidence, evidence="e")


def _card(headline: str = "C1 missing: U1 loses its supply decoupling", severity: str = "MINOR", **changes: object) -> Consequence:
    fields = {
        "headline": headline,
        "consequence": "The VCC net has less decoupling at U1.",
        "severity": severity,
        "safe_to_power": True,
        "fix": "Fit C1.",
        "check_first": "Measure net VCC.",
        "facts_used": ["U1.8 on net VCC"],
        "uncertain": "",
    }
    # ** merges the two dicts into keyword arguments, with the later values winning.
    return Consequence(**{**fields, **changes})


def _fake_client(tmp_path: Path, replies: Callable[[dict], str]) -> tuple[LLMClient, list[dict]]:
    """Build a client that answers from a function, recording each request body."""
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": replies(body)}}],
            },
        )

    settings = LLMSettings(
        base_url="https://llm.test/v1",
        model="test-model",
        api_key="test-key",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=100,
        timeout_s=5,
        cache_dir=tmp_path / "cache",
        reasoning=CFG["llm"]["reasoning"],
    )
    return LLMClient(settings, http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0), requests


# Stage 4


def test_confident_ok_drops_the_suspect() -> None:
    assert fuse(_suspect(), _verdict("OK", 0.7), VERDICT_CFG) is None


def test_unconfident_ok_or_unsure_needs_review_with_the_cv_type() -> None:
    for kind, confidence in (("OK", 0.69), ("UNSURE", 0.9)):
        fault = fuse(_suspect(finding="SHIFTED"), _verdict(kind, confidence), VERDICT_CFG)
        assert (fault.status, fault.fault_type) == ("needs_review", "SHIFTED")


def test_confident_fault_verdict_uses_the_model_type() -> None:
    fault = fuse(_suspect(finding="WRONG_PART"), _verdict("MISSING", 0.6), VERDICT_CFG)
    assert (fault.status, fault.fault_type) == ("confirmed", "MISSING")


def test_unconfident_fault_verdict_keeps_the_cv_type() -> None:
    fault = fuse(_suspect(finding="WRONG_PART"), _verdict("MISSING", 0.59), VERDICT_CFG)
    assert (fault.status, fault.fault_type) == ("confirmed", "WRONG_PART")


def test_prompt_values_for_an_extra_part_read_as_unknown() -> None:
    values = prompt_values(_suspect(None, "EXTRA"), None, NETLIST, CFG)
    assert values["ref"] == "unknown part" and values["polarity_hint"] == "none"
    assert values["best_rotation_deg"] == 180  # A whole number, so the prompt does not read 180.00 degrees.


def test_failed_reply_becomes_unsure_and_the_run_continues(tmp_path: Path) -> None:
    composite = tmp_path / "c.png"
    composite.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    client, requests = _fake_client(tmp_path, lambda body: "not json at all")
    profile = GoldenProfile(board_id="flasher", canonical_size=(100, 100), parts=[])

    judged = judge_suspects(client, [_suspect(composite=str(composite))], profile, NETLIST, CFG)

    assert len(requests) == 2  # The one retry the rules allow.
    assert judged[0].verdict.verdict == "UNSURE"
    assert judged[0].fault.status == "needs_review"


# Stage 5


def test_severity_floors_follow_the_rules() -> None:
    assert severity_floor("REVERSED", "electrolytic_capacitor") == "CRITICAL"
    assert severity_floor("REVERSED", "ic") == "CRITICAL"
    assert severity_floor("ROTATED", "ic") == "CRITICAL"
    assert severity_floor("REVERSED", "diode") == "MAJOR"
    assert severity_floor("MISSING", "electrolytic_capacitor") == "MINOR"


def test_card_cannot_go_below_the_floor_and_critical_is_never_safe() -> None:
    held = apply_floor(_card(severity="MINOR", safe_to_power=True), "CRITICAL")
    assert (held.severity, held.safe_to_power) == ("CRITICAL", False)


def test_card_may_raise_severity_above_the_floor() -> None:
    assert apply_floor(_card(severity="MAJOR"), "MINOR").severity == "MAJOR"


def test_check_card_accepts_facts_from_the_netlist() -> None:
    assert check_card(_card(), "C1", NETLIST) is None


def test_check_card_needs_the_headline_to_start_with_the_ref() -> None:
    assert "must start" in check_card(_card("Missing C1 means trouble"), "C1", NETLIST)
    assert "must start" in check_card(_card("C11 is gone"), "C1", NETLIST)  # A longer ref is not this ref.


def test_check_card_rejects_an_invented_ref() -> None:
    problem = check_card(_card(consequence="This stresses Q9 badly."), "C1", NETLIST)
    assert "Q9" in problem


def test_check_card_rejects_an_invented_pad_and_net() -> None:
    assert "U1.3" in check_card(_card(facts_used=["U1.3 on net VCC"]), "C1", NETLIST)
    assert "RAIL5" in check_card(_card(consequence="This drops net RAIL5."), "C1", NETLIST)


def test_check_card_accepts_eagle_pad_names() -> None:
    netlist = NETLIST.model_copy(update={"nets": [*NETLIST.nets, Net(name="RESET", nodes=["R1.P$2"])]})
    assert check_card(_card(facts_used=["R1.P$2 on net RESET"]), "C1", netlist) is None
    assert "R1.P$9" in check_card(_card(facts_used=["R1.P$9 on net RESET"]), "C1", netlist)


def test_check_card_allows_chip_names_and_ordinary_words_about_nets() -> None:
    assert check_card(_card(consequence="The NE555 keeps running, and the net effect is noise."), "C1", NETLIST) is None


def test_bad_card_is_retried_with_the_error_then_replaced_by_the_template(tmp_path: Path) -> None:
    bad = _card("Invented headline").model_dump_json()
    client, requests = _fake_client(tmp_path, lambda body: bad)
    fault = Fault(ref="C1", fault_type="REVERSED", status="confirmed", suspect=_suspect("C1", "REVERSED"), verdict=_verdict("REVERSED"), consequence=None)

    card = write_card(client, fault, NETLIST, CFG)

    assert len(requests) == 2
    assert "must start" in requests[1]["messages"][1]["content"][0]["text"]
    assert card.headline.startswith("C1 reversed")
    assert (card.severity, card.safe_to_power) == ("CRITICAL", False)
    assert "no valid reply" in card.uncertain


def test_prompt_d_sees_the_pin_swap_table_only_for_orientation_faults(tmp_path: Path) -> None:
    good = _card("U1 reversed: do not power").model_dump_json()
    client, requests = _fake_client(tmp_path, lambda body: good)
    fault = Fault(ref="U1", fault_type="REVERSED", status="confirmed", suspect=_suspect("U1", "REVERSED"), verdict=_verdict("REVERSED"), consequence=None)

    card = write_card(client, fault, NETLIST, CFG)

    text = requests[0]["messages"][1]["content"][0]["text"]
    assert "pin 8 VCC -> pad 4" in text
    assert "not an orientation fault" not in text
    assert card.severity == "CRITICAL"  # The reply said MINOR, and the rule floor wins.


def test_extra_part_and_board_without_a_netlist_use_the_template_without_a_call(tmp_path: Path) -> None:
    client, requests = _fake_client(tmp_path, lambda body: "unused")
    extra = Fault(ref=None, fault_type="EXTRA", status="confirmed", suspect=_suspect(None, "EXTRA"), verdict=_verdict("EXTRA"), consequence=None)
    missing = Fault(ref="P01", fault_type="MISSING", status="confirmed", suspect=_suspect("P01"), verdict=_verdict("MISSING"), consequence=None)

    faults = add_consequences(client, [extra, missing], None, CFG)

    assert requests == []
    assert faults[0].consequence.severity == "MAJOR"
    assert faults[1].consequence.headline.startswith("P01 missing")


def test_template_card_names_the_nets_it_can_prove() -> None:
    card = template_card("C1", "MISSING", "MINOR", NETLIST, "reason")
    assert "GND, VCC" in card.consequence and card.uncertain == "reason"


# Stage 6


def _fault(ref: str | None, severity: str | None, status: str = "confirmed", composite: str = "") -> Fault:
    card = _card(f"{ref} fault", severity=severity, safe_to_power=severity != "CRITICAL") if severity else None
    return Fault(ref=ref, fault_type="MISSING", status=status, suspect=_suspect(ref, composite=composite), verdict=_verdict("MISSING"), consequence=card)


def test_faults_sort_by_severity_then_ref_with_review_last() -> None:
    faults = [_fault("R2", "MINOR"), _fault("C2", "CRITICAL"), _fault("A1", "MAJOR", "needs_review"), _fault("C1", "CRITICAL")]
    assert [f.ref for f in sort_faults(faults)] == ["C1", "C2", "R2", "A1"]


def test_colours_follow_severity_and_review() -> None:
    assert fault_colour(_fault("C1", "CRITICAL")) == (220, 30, 30)
    assert fault_colour(_fault("C1", "CRITICAL", "needs_review")) == fault_colour(_fault("C1", "MINOR"))


def test_only_a_confirmed_critical_fault_makes_a_board_unsafe() -> None:
    assert not is_safe_to_power([_fault("C1", "CRITICAL")])
    assert is_safe_to_power([_fault("C1", "CRITICAL", "needs_review")])
    assert is_safe_to_power([_fault("R1", "MINOR")])
    assert is_safe_to_power([])


def _report(faults: list[Fault]) -> Report:
    return Report(
        board_id="flasher",
        test_image="test.jpg",
        safe_to_power=is_safe_to_power(faults),
        faults=sort_faults(faults),
        timings_s={"align": 1.0},
        versions={"config": "abc"},
    )


def test_banner_is_red_for_a_critical_fault_and_amber_otherwise() -> None:
    assert banner_text(_report([_fault("C1", "CRITICAL")])) == ("Safe to power: NO", "no")
    assert banner_text(_report([_fault("R1", "MINOR")]))[1] == "check"
    assert banner_text(_report([]))[1] == "check"


def test_overlay_draws_through_the_inverse_homography() -> None:
    photo = np.zeros((200, 300, 3), np.uint8)
    # This warp moves the photo 100 px right into the canonical frame, so canonical x maps back 100 px left.
    homography = np.array([[1.0, 0, 100], [0, 1, 0], [0, 0, 1]])
    fault = _fault("C1", "CRITICAL")
    fault.suspect.bbox = (110, 40, 150, 80)

    overlay = draw_report_overlay(photo, [fault], homography, 2400)

    assert tuple(overlay[60, 10]) == (220, 30, 30)  # The box's left edge, at x = 110 - 100.
    assert tuple(overlay[60, 110]) == (0, 0, 0)     # Nothing at the canonical position.


def test_overlay_shrinks_a_large_photo_to_the_limit() -> None:
    overlay = draw_report_overlay(np.zeros((1000, 4000, 3), np.uint8), [], np.eye(3), 2000)
    assert overlay.shape[:2] == (500, 2000)


def test_report_files_are_written_and_the_page_is_self_contained(tmp_path: Path) -> None:
    composite = tmp_path / "composite.png"
    cv2.imwrite(str(composite), np.full((50, 100, 3), 128, np.uint8))
    report = _report([_fault("C1", "CRITICAL", composite=str(composite))])

    paths = save_report(report, np.zeros((200, 300, 3), np.uint8), np.eye(3), tmp_path, CFG)

    page = paths["report.html"].read_text(encoding="utf-8")
    assert "Safe to power: NO" in page and "C1 fault" in page
    assert page.count("data:image/jpeg;base64,") == 2  # The overlay and the card's crops, both embedded.
    assert 'prefers-color-scheme: dark' in page
    assert Report.model_validate_json(paths["report.json"].read_text(encoding="utf-8")).board_id == "flasher"
    assert paths["overlay.png"].exists()


def test_html_escapes_model_text(tmp_path: Path) -> None:
    fault = _fault("C1", "MINOR")
    fault.consequence.consequence = "<script>alert(1)</script>"
    paths = save_report(_report([fault]), np.zeros((50, 50, 3), np.uint8), np.eye(3), tmp_path, CFG)
    assert "<script>" not in paths["report.html"].read_text(encoding="utf-8")


# Evaluation


def _prepared(ok: bool = True) -> PreparedPhoto:
    return PreparedPhoto(ok, "Aligned." if ok else "Retake.", np.zeros((100, 200, 3), np.uint8), np.eye(3), [], 1.0, 1.0, 100, 1.0, True, {"align": 1.0})


def _board(tmp_path: Path) -> GoldenBoard:
    from pocket_aoi.align import GoldenFrame

    image = np.zeros((100, 200, 3), np.uint8)
    frame = GoldenFrame(image, np.ones((100, 200), bool), np.eye(3), (200, 100))
    profile = GoldenProfile(board_id="b", canonical_size=(200, 100), parts=[])
    return GoldenBoard("b", profile, image, mask_to_rle(np.ones((100, 200), bool)), frame, None)


def test_clean_sample_keeps_defect_photos_and_only_grows_as_it_gets_larger() -> None:
    defect = GroundTruth(photo="a/defect.jpg", board_id="a", defects=[GroundTruthDefect(ref="C1", type="MISSING")])
    clean = [GroundTruth(photo=f"{board}/{i:02}.jpg", board_id=board, defects=[]) for board in "ab" for i in range(20)]
    records = [defect, *clean]

    small = sample_clean_photos(records, 3, seed=1)
    large = sample_clean_photos(records, 8, seed=1)

    assert defect in small
    assert sum(r.board_id == "a" and not r.defects for r in small) == 3
    assert sum(r.board_id == "b" and not r.defects for r in small) == 3
    assert {r.photo for r in small} <= {r.photo for r in large}
    assert small == sample_clean_photos(records, 3, seed=1)
    assert small == [r for r in records if r in small]
    assert len(sample_clean_photos(records, 99, seed=1)) == len(records)


def test_marker_hint_reaches_the_prompt_and_defaults_to_unknown() -> None:
    plain = Part(ref="D1", cls="diode", bbox=(0, 0, 9, 9), centroid=(4, 4), mask_rle={}, score=1.0, package=None, polarized=True)
    hinted = plain.model_copy(update={"marker_hint": "band at the bottom end"})
    assert prompt_values(_suspect("D1"), plain, None, CFG)["golden_marker_hint"] == "unknown"
    assert prompt_values(_suspect("D1"), hinted, None, CFG)["golden_marker_hint"] == "band at the bottom end"


def test_banner_never_calls_a_board_safe() -> None:
    assert "safe" not in banner_text(_report([]))[0].lower()
    assert "by eye" in banner_text(_report([]))[0]
    assert "by eye" in banner_text(_report([_fault("R1", "MINOR")]))[0]


def test_result_sheet_shows_both_crops_and_fits_every_tile() -> None:
    golden = np.zeros((300, 400, 3), np.uint8)
    test = np.full((300, 400, 3), 200, np.uint8)
    tile = make_tile(golden, test, (100, 100, 200, 150), "pcb1_000 R1 MISSING")
    middle = tile.shape[0] // 2
    assert tile[middle, tile.shape[1] // 4].tolist() == [0, 0, 0]  # The golden crop is on the left.
    assert tile[middle, 3 * tile.shape[1] // 4].tolist() == [200, 200, 200]

    sheet = make_sheet([tile] * 5, columns=2)
    assert sheet.shape[0] >= 3 * tile.shape[0]
    assert sheet.shape[1] >= 2 * tile.shape[1]


def test_polarity_eval_prefers_an_unaugmented_copy_of_each_shot() -> None:
    records = [
        {"id": "b", "name": "one.jpg", "split": "train", "boxes": []},
        {"id": "a", "name": "one.jpg", "split": "test", "boxes": []},
        {"id": "c", "name": "two.jpg", "split": "train", "boxes": []},
    ]
    shots = pick_shots(records)
    assert shots["one.jpg"]["id"] == "a" and shots["two.jpg"]["id"] == "c"


def test_polarity_eval_reads_a_parts_state_from_labels_of_its_own_kind() -> None:
    box = (100.0, 100.0, 140.0, 200.0)
    centre = {"x": 120, "y": 150, "width": 40, "height": 100}
    assert label_box(centre) == box
    assert status_in_shot({"boxes": [{"label": "Diode Opposite polarity", **centre}]}, box, "D1") == "REVERSED"
    assert status_in_shot({"boxes": [{"label": "Diode-missed-welding", **centre}]}, box, "D1") == "MISSING"
    assert status_in_shot({"boxes": [{"label": "R-missed-welding", **centre}]}, box, "D1") == "OK"
    assert status_in_shot({"boxes": [{"label": "Diode Opposite polarity", "x": 400, "y": 400, "width": 40, "height": 100}]}, box, "D1") == "OK"
    assert status_in_shot({"boxes": [{"label": "Diode Opposite polarity", **centre}]}, box, "E1") == "OK"


def test_share_in_frame_counts_pixels_landing_inside() -> None:
    mask = np.zeros((100, 200), bool)
    mask[0:10, 190:200] = True
    assert share_in_frame(mask, np.eye(3), (200, 100)) == 1.0
    moved = np.array([[1.0, 0, 5], [0, 1, 0], [0, 0, 1]])  # Half the defect's columns slide off the right side.
    assert share_in_frame(mask, moved, (200, 100)) == pytest.approx(0.5)
    assert share_in_frame(np.zeros((10, 10), bool), np.eye(3), (10, 10)) == 0.0


def _write_mask(tmp_path: Path, box: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    mask = np.zeros(shape, np.uint8)
    mask[box[1] : box[3], box[0] : box[2]] = 255
    path = tmp_path / "mask.png"
    cv2.imwrite(str(path), mask)
    return str(path)  # An absolute path overrides the repo root when the two are joined.


def test_score_counts_a_found_defect_and_the_claims_elsewhere_as_extra(tmp_path: Path) -> None:
    mask = _write_mask(tmp_path, (10, 10, 40, 40), (100, 200))
    truth = GroundTruth(photo="p.jpg", board_id="b", defects=[GroundTruthDefect(type="MISSING", mask=mask)])
    claims = [Claim("P1", "MISSING", (5, 5, 45, 45), {}), Claim("P2", "SHIFTED", (150, 50, 190, 90), {})]

    result = score_photo(truth, _prepared(), _board(tmp_path), claims, [], CFG["eval"])

    assert (result.planted, result.found_any, result.found_right_type, result.extra_findings) == (1, 1, 1, 1)


def test_score_keeps_a_defect_off_the_board_apart_from_the_planted_count(tmp_path: Path) -> None:
    # The whole mask lands outside the canonical frame once the homography moves it.
    mask = _write_mask(tmp_path, (10, 10, 40, 40), (100, 200))
    truth = GroundTruth(photo="p.jpg", board_id="b", defects=[GroundTruthDefect(type="SHIFTED", mask=mask)])
    prepared = PreparedPhoto(True, "Aligned.", np.zeros((100, 200, 3), np.uint8), np.array([[1.0, 0, -100], [0, 1, 0], [0, 0, 1]]), [], 1.0, 1.0, 100, 1.0, True, {})

    result = score_photo(truth, prepared, _board(tmp_path), [], [], CFG["eval"])

    assert (result.planted, result.off_board, result.missed) == (0, 1, [])


def test_score_records_a_defect_seen_only_in_review(tmp_path: Path) -> None:
    mask = _write_mask(tmp_path, (10, 10, 40, 40), (100, 200))
    truth = GroundTruth(photo="p.jpg", board_id="b", defects=[GroundTruthDefect(type="MISSING", mask=mask)])

    result = score_photo(truth, _prepared(), _board(tmp_path), [], [Claim("P1", "MISSING", (5, 5, 45, 45), {})], CFG["eval"])

    assert (result.found_any, result.found_in_review, result.needs_review) == (0, 1, 1)


def test_rejected_photo_counts_its_defects_as_missed(tmp_path: Path) -> None:
    truth = GroundTruth(photo="p.jpg", board_id="b", defects=[GroundTruthDefect(type="MISSING", ref="C1")])
    result = score_photo(truth, _prepared(ok=False), _board(tmp_path), [], [], CFG["eval"])
    assert (result.planted, result.found_any, result.missed) == (1, 0, ["MISSING"])


def test_summary_by_label_ignores_other_labels_on_the_same_photo() -> None:
    results = [
        PhotoResult("a", "b", "melt,missing", "dev", True, planted=1, found_any=1, found_right_type=1),
        PhotoResult("c", "b", "normal", "dev", True),
    ]
    summary = summarise_by_label(results)
    assert summary["missing"]["found_any_type"] == 1
    assert summary["clean"]["clean_photos"] == 1


def test_ablation_table_compares_both_pipelines() -> None:
    cv = [PhotoResult("a", "b", "normal", "dev", True, extra_findings=4)]
    fused = [PhotoResult("a", "b", "normal", "dev", True, extra_findings=1)]
    table = format_ablation(cv, fused)
    assert "| CV only | none | none | 0 | 4.00 |" in table
    assert "| CV + Muse Glimmer | none | none | 0 | 1.00 |" in table


def test_unsure_verdict_validates_as_a_verdict() -> None:
    assert unsure_verdict("no reply").confidence == 0.0


# Prompt B


def test_silkscreen_accepts_only_a_ref_on_the_board_with_a_fitting_prefix() -> None:
    from pocket_aoi.silkscreen import accept_ref

    candidates = ["C1", "R1", "U1", "LED1", "S1"]
    assert accept_ref("C1", "electrolytic_capacitor", candidates) == "C1"
    assert accept_ref("LED1", "led", candidates) == "LED1"
    assert accept_ref("S1", "other", candidates) == "S1"     # "other" has no prefix rule.
    assert accept_ref("R9", "resistor", candidates) is None  # Not on this board.
    assert accept_ref("C1", "resistor", candidates) is None  # Wrong prefix for the class.
    assert accept_ref(None, "resistor", candidates) is None


def test_silkscreen_blanks_a_ref_read_twice_and_needs_a_netlist(tmp_path: Path) -> None:
    from pocket_aoi.silkscreen import read_refs

    image = np.zeros((200, 200, 3), np.uint8)
    box = (40, 40, 80, 80)
    mask = np.zeros((200, 200), bool)
    mask[40:80, 40:80] = True
    parts = [Part(ref=None, cls="resistor", bbox=box, centroid=(60.0, 60.0), mask_rle=mask_to_rle(mask), score=0.9, package=None, polarized=False)] * 2
    reply = json.dumps({"ref": "R1", "alternatives": [], "confidence": 0.9})
    client, _ = _fake_client(tmp_path, lambda body: reply)

    assert read_refs(client, image, parts, NETLIST, CFG) == [None, None]
    assert read_refs(client, image, parts[:1], NETLIST, CFG) == ["R1"]
    assert read_refs(client, image, parts, None, CFG) == [None, None]


# Demo defects


def _demo_part() -> Part:
    mask = np.zeros((100, 200), bool)
    mask[20:60, 40:120] = True
    return Part(ref="C1", cls="electrolytic_capacitor", bbox=(40, 20, 120, 60), centroid=(80.0, 40.0), mask_rle=mask_to_rle(mask), score=0.9, package=None, polarized=True)


def test_reverse_part_turns_only_the_part_by_half_a_turn() -> None:
    from eval.make_demo_defects import reverse_part

    rng = np.random.default_rng(0)
    canonical = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
    edited, changed = reverse_part(canonical, _demo_part())
    assert np.array_equal(edited[20:60, 40:120], canonical[20:60, 40:120][::-1, ::-1])
    assert np.array_equal(edited[:20], canonical[:20])  # Nothing outside the box moved.
    assert changed.sum() == 40 * 80


def test_remove_part_paints_over_it_and_marks_a_margin() -> None:
    from eval.make_demo_defects import remove_part

    canonical = np.full((100, 200, 3), (30, 110, 60), np.uint8)
    canonical[20:60, 40:120] = (240, 240, 240)
    edited, changed = remove_part(canonical, _demo_part())
    assert edited[40, 80].max() < 200   # The bright part is gone.
    assert changed[18, 80] and not changed[5, 80]  # A small margin, not the whole board.


def test_paste_back_changes_only_the_edited_area_of_the_photo() -> None:
    from eval.make_demo_defects import paste_back

    photo = np.full((100, 200, 3), 50, np.uint8)
    edited = np.full((100, 200, 3), 200, np.uint8)
    changed = np.zeros((100, 200), bool)
    changed[40:60, 80:120] = True
    homography = np.array([[1.0, 0, 10], [0, 1, 0], [0, 0, 1]])  # Canonical x is photo x plus 10.

    result = paste_back(photo, edited, changed, homography)

    assert result[50, 90].mean() > 150   # Canonical x = 100, inside the edit.
    assert result[50, 150].mean() == 50  # Far outside it, untouched.
