"""Load configs/default.yaml and .env, and hash the config so each report can record it."""

import hashlib
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# parents[2] walks up from src/pocket_aoi/config.py to the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read a YAML config into nested dicts and put the values from .env in the environment."""
    load_dotenv(REPO_ROOT / ".env")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def config_hash(path: Path = DEFAULT_CONFIG) -> str:
    """Return a short SHA-256 of the config file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
