"""OpenAI-compatible client for Muse Glimmer: prompt files, JSON validation and a call cache.

All Glimmer calls in the project go through this module (AGENTS.md section 5, rule 3).
"""

import base64
import hashlib
import json
import logging
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import httpx
from openai import APIError, AuthenticationError, BadRequestError, OpenAI, PermissionDeniedError
from pydantic import BaseModel, ValidationError

from pocket_aoi.config import REPO_ROOT

logger = logging.getLogger(__name__)

PROMPTS_DIR = REPO_ROOT / "prompts"

# Matches {ref} but not the {"ref": ...} in a JSON example, so literal braces survive.
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

_OUT_OF_CREDITS = 402
_MAX_FAILURES_IN_A_ROW = 5

# Like a C++ template parameter: complete_json returns the same model type it is given.
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class Prompt:
    """One prompts/<name>_<version>.md file, split into its two messages."""

    name: str
    version: str
    system: str
    user: str

    @property
    def id(self) -> str:
        return f"{self.name}_{self.version}"


def load_prompt(name: str, version: str) -> Prompt:
    """Read a prompt file laid out as '# System', its text, '# User', its text."""
    text = (PROMPTS_DIR / f"{name}_{version}.md").read_text(encoding="utf-8")
    # partition returns (before, separator, after); _ is the convention for an unused value.
    _, _, rest = text.partition("# System\n")
    system, _, user = rest.partition("# User\n")
    return Prompt(name, version, system.strip(), user.strip())


def render(template: str, values: Mapping[str, object]) -> str:
    """Fill the named {placeholders}, formatting floats to two decimals."""

    # A nested function can read values from the enclosing call, like a C++ lambda capturing by reference.
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"prompt placeholder {{{key}}} has no value")
        value = values[key]
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    return _PLACEHOLDER.sub(replace, template)


def extract_json(reply: str) -> str:
    """Return the outermost {...} span, since models sometimes wrap JSON in prose or code fences."""
    start, end = reply.find("{"), reply.rfind("}")
    return reply[start : end + 1] if start != -1 and end > start else reply


def _image_data_url(image: bytes) -> str:
    """Encode PNG or JPEG bytes as the data URL that OpenAI-compatible vision APIs accept."""
    if image.startswith(b"\x89PNG"):
        mime = "image/png"
    elif image.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    else:
        raise ValueError("image must be PNG or JPEG bytes")
    return f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    model: str
    api_key: str = field(repr=False)  # Left out of repr so a logged settings object cannot leak the key.
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    timeout_s: float
    cache_dir: Path
    reasoning: Mapping[str, str]

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "LLMSettings":
        """Combine the backend from .env with the sampling settings from the YAML config."""
        missing = [n for n in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY") if not os.environ.get(n)]
        if missing:
            raise ValueError(f"{', '.join(missing)} not set. Copy .env.example to .env and fill it in.")
        llm = cfg["llm"]
        return cls(
            base_url=os.environ["LLM_BASE_URL"],
            model=os.environ["LLM_MODEL"],
            api_key=os.environ["LLM_API_KEY"],
            temperature=llm["temperature"],
            top_p=llm["top_p"],
            top_k=llm["top_k"],
            max_tokens=llm["max_tokens"],
            timeout_s=llm["timeout_s"],
            cache_dir=REPO_ROOT / llm["cache_dir"],
            reasoning=llm["reasoning"],
        )


class LLMClient:
    """Sends prompts to the configured backend and caches every reply on disk."""

    def __init__(
        self,
        settings: LLMSettings,
        http_client: httpx.Client | None = None,
        max_retries: int = 4,
    ) -> None:
        self.settings = settings
        self._openai = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_s,
            max_retries=max_retries,
            http_client=http_client,
        )
        self.use_response_format = True
        self._failures_in_a_row = 0

    def complete(
        self,
        prompt: Prompt,
        values: Mapping[str, object] | None = None,
        *,  # Everything after this bare * must be passed by name.
        image: bytes | None = None,
        schema: type[BaseModel] | None = None,
        use_cache: bool = True,
        feedback: str = "",
    ) -> str:
        """Return the model's reply text, from the cache when this exact call was made before."""
        # The chat template defaults to high when this line is absent, and it reads the line on any backend.
        system = f"{prompt.system}\n\nReasoning strength: {self.settings.reasoning[prompt.name]}."
        user = render(prompt.user, values or {})
        if feedback:
            user = f"{user}\n\n{feedback}"

        cache_path = self._cache_path(prompt, system, user, image)
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["reply"]

        reply = self._call(system, user, image, schema)

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"prompt": prompt.id, "model": self.settings.model, "reply": reply}
            cache_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return reply

    def complete_json(
        self,
        schema: type[SchemaT],
        prompt: Prompt,
        values: Mapping[str, object] | None = None,
        *,
        image: bytes | None = None,
        use_cache: bool = True,
        check: Callable[[SchemaT], str | None] | None = None,
    ) -> SchemaT | None:
        """Return a validated reply, retrying once with the error appended.

        Returns None if the reply stays invalid or the call fails, so the caller records
        UNSURE and the run continues. A rejected API key, an empty account or a backend that
        keeps failing still raises, because every later call would fail too, marking every
        part on the board UNSURE and quietly corrupting an evaluation. An optional check adds
        rules that a schema cannot express, returning an error message or None.
        """
        feedback = ""
        for _ in range(2):
            try:
                reply = self.complete(
                    prompt, values, image=image, schema=schema, use_cache=use_cache, feedback=feedback
                )
                self._failures_in_a_row = 0
                parsed = schema.model_validate_json(extract_json(reply))
                problem = check(parsed) if check is not None else None
                if problem is None:
                    return parsed
                logger.warning("%s reply failed its check: %s", prompt.id, problem)
                feedback = f"Your previous reply was rejected:\n{problem}\nReturn only the corrected JSON object."
            except ValidationError as error:
                logger.warning("%s reply failed validation: %s", prompt.id, error)
                feedback = (
                    f"Your previous reply was rejected by the validator:\n{error}\n"
                    "Return only the corrected JSON object."
                )
            except (AuthenticationError, PermissionDeniedError):
                raise
            except APIError as error:
                # getattr with a default, because a connection error has no HTTP status at all.
                self._failures_in_a_row += 1
                if getattr(error, "status_code", None) == _OUT_OF_CREDITS or self._failures_in_a_row >= _MAX_FAILURES_IN_A_ROW:
                    raise
                logger.warning("%s call failed: %s", prompt.id, error)
                return None
        return None

    def _call(self, system: str, user: str, image: bytes | None, schema: type[BaseModel] | None) -> str:
        """Make one API request and return the reply text."""
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        if image is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": _image_data_url(image)}})

        request: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
            "extra_body": {"top_k": self.settings.top_k},  # top_k is not a standard OpenAI field.
        }
        if schema is not None and self.use_response_format:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }

        try:
            # **request expands the dict into keyword arguments.
            response = self._openai.chat.completions.create(**request)
        except BadRequestError:
            if "response_format" not in request:
                raise
            # Not every backend accepts a JSON schema. The prompt still asks for JSON, and pydantic still checks it.
            del request["response_format"]
            response = self._openai.chat.completions.create(**request)
            self.use_response_format = False
        return response.choices[0].message.content or ""

    def _cache_path(self, prompt: Prompt, system: str, user: str, image: bytes | None) -> Path:
        """Hash the image, prompt version, model id and rendered text into a cache file path.

        The rendered text is in the key because Prompt D has no image, so every consequence
        card would otherwise share one entry.
        """
        digest = hashlib.sha256(image or b"")
        digest.update(json.dumps([prompt.id, self.settings.model, system, user]).encode("utf-8"))
        return self.settings.cache_dir / f"{digest.hexdigest()}.json"
