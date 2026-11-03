"""Exercise llm.py against a fake HTTP backend, so the real request bytes are checked with no network."""

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from openai import APIStatusError, AuthenticationError
from pydantic import BaseModel

from pocket_aoi.config import load_config
from pocket_aoi.llm import LLMClient, LLMSettings, Prompt, extract_json, load_prompt, render
from pocket_aoi.schemas import Verdict

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"
OTHER_PNG_BYTES = b"\x89PNG\r\n\x1a\nother"
VALID_VERDICT = json.dumps(
    {
        "golden_observation": "stripe left",
        "test_observation": "stripe right",
        "verdict": "REVERSED",
        "confidence": 0.9,
        "evidence": "the stripe moved",
    }
)


class Answer(BaseModel):
    value: int


def _chat_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        },
    )


def _client(
    tmp_path: Path, handler: Callable[[httpx.Request], httpx.Response]
) -> tuple[LLMClient, list[dict]]:
    """Build a client whose HTTP traffic goes to handler, and record each request body."""
    requests: list[dict] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return handler(request)

    settings = LLMSettings(
        base_url="https://backend.test/v1",
        model="test-model",
        api_key="test-key",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=100,
        timeout_s=5,
        cache_dir=tmp_path / "cache",
        reasoning={"verdict": "low", "consequence": "medium"},
    )
    http_client = httpx.Client(transport=httpx.MockTransport(recording_handler))
    return LLMClient(settings, http_client=http_client, max_retries=0), requests


VERDICT_PROMPT = Prompt("verdict", "v1", "You inspect boards.", "Part {ref}, similarity {ssim}.")
CONSEQUENCE_PROMPT = Prompt("consequence", "v1", "You explain faults.", "Fault on {ref}.")


def test_render_fills_placeholders_and_keeps_json_braces() -> None:
    template = 'Part {ref} at {ssim}. Return {"ref": "R3", "alternatives": []}'
    assert render(template, {"ref": "C7", "ssim": 0.41234}) == (
        'Part C7 at 0.41. Return {"ref": "R3", "alternatives": []}'
    )


def test_render_rejects_a_missing_value() -> None:
    with pytest.raises(KeyError, match="ref"):
        render("Part {ref}", {})


@pytest.mark.parametrize(
    ("name", "placeholders"),
    [
        ("verdict", {"ref", "part_class", "package", "value", "polarity_hint", "golden_marker_hint",
                     "cv_finding", "ssim", "best_rotation_deg", "offset_ratio"}),
        ("silkscreen", {"part_class", "candidate_refs"}),
        ("netlist", {"netlist_schema_example"}),
        ("consequence", {"board_description", "supply_volts", "supply_net", "ref", "fault_type", "confidence",
                         "value", "part_type", "package", "pin_net_table", "neighbour_table", "pin_swap_table",
                         "severity_floor"}),
    ],
)
def test_prompt_files_load_with_expected_placeholders(name: str, placeholders: set[str]) -> None:
    prompt = load_prompt(name, "v1")
    assert prompt.system
    assert render(prompt.user, dict.fromkeys(placeholders, "x"))  # Every placeholder is accounted for.


def test_every_prompt_has_a_reasoning_level_in_the_config() -> None:
    cfg = load_config()
    assert set(cfg["llm"]["prompt_versions"]) == set(cfg["llm"]["reasoning"])


def test_extract_json_strips_code_fences_and_prose() -> None:
    assert extract_json('Sure:\n```json\n{"value": 1}\n```\nDone.') == '{"value": 1}'


def test_request_carries_image_sampling_schema_and_reasoning(tmp_path: Path) -> None:
    client, requests = _client(tmp_path, lambda _: _chat_response(VALID_VERDICT))
    client.complete_json(Verdict, VERDICT_PROMPT, {"ref": "C1", "ssim": 0.5}, image=PNG_BYTES)

    body = requests[0]
    assert body["model"] == "test-model"
    assert (body["temperature"], body["top_p"], body["top_k"]) == (1.0, 0.95, 64)
    assert body["messages"][0]["content"].endswith("Reasoning strength: low.")
    image_part, text_part = body["messages"][1]["content"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert text_part["text"] == "Part C1, similarity 0.50."
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "Verdict"


def test_reasoning_level_follows_the_prompt_name(tmp_path: Path) -> None:
    client, requests = _client(tmp_path, lambda _: _chat_response("ok"))
    client.complete(CONSEQUENCE_PROMPT, {"ref": "C1"})
    assert requests[0]["messages"][0]["content"].endswith("Reasoning strength: medium.")


def test_identical_call_is_served_from_cache(tmp_path: Path) -> None:
    client, requests = _client(tmp_path, lambda _: _chat_response(VALID_VERDICT))
    values = {"ref": "C1", "ssim": 0.5}
    first = client.complete_json(Verdict, VERDICT_PROMPT, values, image=PNG_BYTES)
    second = client.complete_json(Verdict, VERDICT_PROMPT, values, image=PNG_BYTES)
    assert first == second
    assert len(requests) == 1


def test_a_different_image_misses_the_cache(tmp_path: Path) -> None:
    client, requests = _client(tmp_path, lambda _: _chat_response(VALID_VERDICT))
    values = {"ref": "C1", "ssim": 0.5}
    client.complete_json(Verdict, VERDICT_PROMPT, values, image=PNG_BYTES)
    client.complete_json(Verdict, VERDICT_PROMPT, values, image=OTHER_PNG_BYTES)
    assert len(requests) == 2


def test_text_only_prompts_do_not_collide_in_the_cache(tmp_path: Path) -> None:
    """Prompt D has no image, so two different faults must still get two different cards."""
    client, requests = _client(tmp_path, lambda _: _chat_response("card"))
    client.complete(CONSEQUENCE_PROMPT, {"ref": "C1"})
    client.complete(CONSEQUENCE_PROMPT, {"ref": "C2"})
    assert len(requests) == 2


def test_invalid_reply_is_retried_with_the_error_appended(tmp_path: Path) -> None:
    replies = iter(['{"value": "not a number"}', '{"value": 7}'])
    client, requests = _client(tmp_path, lambda _: _chat_response(next(replies)))
    prompt = Prompt("verdict", "v1", "s", "u")

    assert client.complete_json(Answer, prompt) == Answer(value=7)
    assert len(requests) == 2
    assert "rejected by the validator" in requests[1]["messages"][1]["content"][0]["text"]


def test_reply_invalid_twice_returns_none_rather_than_raising(tmp_path: Path) -> None:
    client, requests = _client(tmp_path, lambda _: _chat_response("no json here"))
    assert client.complete_json(Answer, Prompt("verdict", "v1", "s", "u")) is None
    assert len(requests) == 2


def test_server_error_returns_none_rather_than_raising(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, lambda _: httpx.Response(500, json={"error": {"message": "down"}}))
    assert client.complete_json(Answer, Prompt("verdict", "v1", "s", "u")) is None


def test_rejected_api_key_raises(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, lambda _: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(AuthenticationError):
        client.complete_json(Answer, Prompt("verdict", "v1", "s", "u"))


def test_backend_without_json_schema_support_falls_back(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "response_format" in json.loads(request.content):
            return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})
        return _chat_response('{"value": 3}')

    client, requests = _client(tmp_path, handler)
    prompt = Prompt("verdict", "v1", "s", "u")
    assert client.complete_json(Answer, prompt) == Answer(value=3)
    assert client.complete_json(Answer, prompt, use_cache=False) == Answer(value=3)
    # Rejected once, retried without it, and never sent again.
    assert ["response_format" in r for r in requests] == [True, False, False]


def test_empty_account_raises_instead_of_reading_as_unsure(tmp_path: Path) -> None:
    client, requests = _client(tmp_path, lambda _: httpx.Response(402, json={"error": {"message": "no credits"}}))
    with pytest.raises(APIStatusError) as caught:
        client.complete_json(Verdict, VERDICT_PROMPT, {"ref": "C7", "ssim": 0.5}, image=PNG_BYTES)
    assert caught.value.status_code == 402
    assert len(requests) == 1  # Retrying cannot help, so it stops at once.


def test_backend_that_keeps_failing_raises_but_one_failure_does_not(tmp_path: Path) -> None:
    status = {"code": 500}

    def handler(_: httpx.Request) -> httpx.Response:
        if status["code"] == 200:
            return _chat_response(VALID_VERDICT)
        return httpx.Response(status["code"], json={"error": {"message": "server gone"}})

    client, _ = _client(tmp_path, handler)
    values = {"ref": "C7", "ssim": 0.5}

    # Four failures in a row each read as "no reply", so the run carries on.
    for number in range(4):
        assert client.complete_json(Verdict, VERDICT_PROMPT, {**values, "ref": f"C{number}"}, image=PNG_BYTES) is None

    # A success resets the count, so scattered failures never stop a run.
    status["code"] = 200
    assert client.complete_json(Verdict, VERDICT_PROMPT, values, image=PNG_BYTES) is not None
    status["code"] = 500
    for number in range(4):
        assert client.complete_json(Verdict, VERDICT_PROMPT, {**values, "ref": f"D{number}"}, image=PNG_BYTES) is None

    # The fifth in a row means the backend is down, so it stops rather than marking everything UNSURE.
    with pytest.raises(APIStatusError):
        client.complete_json(Verdict, VERDICT_PROMPT, {**values, "ref": "E1"}, image=PNG_BYTES)


def test_settings_repr_hides_the_api_key(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, lambda _: _chat_response(""))
    assert "test-key" not in repr(client.settings)
