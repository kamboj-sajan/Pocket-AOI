"""The Gradio page's handlers and the photo download's error text. No network or GPU is needed."""

from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import pytest
import yaml
from openai import APIConnectionError

from app import gradio_app
from eval.fetch_roboflow_pcba import check_reply


def _backend_down(*_args: Any, **_kwargs: Any) -> Any:
    raise APIConnectionError(request=httpx.Request("POST", "https://example.test/v1/chat/completions"))


def test_inspect_says_plainly_when_the_backend_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gradio_app, "get_client", lambda: object())
    monkeypatch.setattr(gradio_app, "load_golden_board", lambda *_: object())
    monkeypatch.setattr(gradio_app, "inspect_photo", _backend_down)
    with pytest.raises(gr.Error, match="did not answer"):
        gradio_app.inspect("flasher", "photo.jpg")


def test_setup_leaves_refs_for_typing_when_the_backend_is_down(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    part = type("FakePart", (), {"ref": None, "cls": "resistor", "bbox": (1, 2, 3, 4), "score": 0.9})
    profile = type("FakeProfile", (), {"parts": [part]})
    board = type("FakeBoard", (), {"netlist": object(), "image": None})
    monkeypatch.setattr(gradio_app, "setup_board", lambda *_: profile)
    monkeypatch.setattr(gradio_app, "load_golden_board", lambda *_: board)
    monkeypatch.setattr(gradio_app, "get_client", lambda: object())
    monkeypatch.setattr(gradio_app, "read_refs", _backend_down)
    monkeypatch.setattr(gradio_app, "_overlay", lambda *_: None)

    _, rows, note = gradio_app.find_parts("flasher", str(tmp_path / "golden.jpg"), None)
    assert rows[0][0] == ""
    assert "Silkscreen reading was skipped" in note


def test_saving_the_table_keeps_marker_hints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    board_dir = tmp_path / "flasher"
    board_dir.mkdir()
    (board_dir / "golden_profile.json").write_text("{}", encoding="utf-8")
    saved = {"parts": [{"ref": "D1", "cls": "diode", "box": [1, 2, 3, 4], "marker_hint": "band at the bottom end"}]}
    (board_dir / "refs.yaml").write_text(yaml.safe_dump(saved), encoding="utf-8")
    monkeypatch.setattr(gradio_app, "BOARDS_DIR", tmp_path)
    monkeypatch.setattr(gradio_app, "relabel_board", lambda *_: None)
    monkeypatch.setattr(gradio_app, "_overlay", lambda *_: None)
    monkeypatch.setattr(gradio_app, "list_boards", lambda: ["flasher"])

    table = [["D1", "diode", 1, 2, 3, 4, 0.9], ["R1", "resistor", 5, 6, 7, 8, 0.8], ["", "other", 9, 9, 9, 9, 0.7]]
    _, note, _ = gradio_app.save_refs("flasher", table)

    labels = yaml.safe_load((board_dir / "refs.yaml").read_text(encoding="utf-8"))["parts"]
    assert labels[0]["marker_hint"] == "band at the bottom end"
    assert "marker_hint" not in labels[1]
    assert "Saved 2 parts" in note


def test_a_failed_download_does_not_print_the_key() -> None:
    request = httpx.Request("GET", "https://api.roboflow.com/x/images/1?api_key=SECRET")
    with pytest.raises(SystemExit) as stopped:
        check_reply(httpx.Response(401, request=request))
    assert "401" in str(stopped.value)
    assert "SECRET" not in str(stopped.value)
