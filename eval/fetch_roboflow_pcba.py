"""Download the PCBA photos and their original labels from Roboflow, for eval/run_polarity_eval.py.

The dataset is PCBA by user 123 on Roboflow Universe, CC BY 4.0:
https://universe.roboflow.com/123-1fark/pcba-lw1r1
Its normal export merges every class into one, so this script reads each image's labels through the API.
It needs a free account's key in .env as ROBOFLOW_API_KEY. Everything lands in data/raw/roboflow_pcba/.

Run from the repo root:
    python -m eval.fetch_roboflow_pcba
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from dotenv import dotenv_values

from pocket_aoi.config import REPO_ROOT

PROJECT_URL = "https://api.roboflow.com/123-1fark/pcba-lw1r1"
OUT_DIR = REPO_ROOT / "data" / "raw" / "roboflow_pcba"
PAGE_SIZE = 250  # The search endpoint's largest page.
WORKERS = 4  # Few, to stay gentle on a free service.


def check_reply(reply: httpx.Response) -> None:
    """Stop on a failed call, naming only the path: httpx's own error text holds the full URL, and so the key."""
    if reply.is_error:
        raise SystemExit(f"Roboflow replied {reply.status_code} for {reply.request.url.path}")


def find_images(key: str) -> list[dict[str, Any]]:
    """List every image in the project, a page at a time."""
    found: list[dict[str, Any]] = []
    while True:
        reply = httpx.post(
            f"{PROJECT_URL}/search",
            params={"api_key": key},
            json={"offset": len(found), "limit": PAGE_SIZE, "in_dataset": True, "fields": ["id", "name", "split"]},
            timeout=120,
        )
        check_reply(reply)
        page = reply.json()["results"]
        found.extend(page)
        if not page or len(found) >= reply.json()["total"]:
            return found


def fetch_record(item: dict[str, Any], key: str) -> dict[str, Any]:
    """Read one image's labels and its link."""
    reply = httpx.get(f"{PROJECT_URL}/images/{item['id']}", params={"api_key": key}, timeout=60)
    check_reply(reply)
    image = reply.json()["image"]
    note = image.get("annotation") or {}
    return {
        "id": image["id"],
        "name": image["name"],
        "split": image.get("split"),
        "width": note.get("width"),
        "height": note.get("height"),
        "boxes": note.get("boxes") or [],
        "original": image["urls"]["original"],
    }


def fetch_photo(record: dict[str, Any]) -> None:
    """Save one source photo, which matches its labels' coordinates exactly."""
    target = OUT_DIR / "source" / f"{record['id']}.jpg"
    if target.exists():
        return
    reply = httpx.get(record["original"], timeout=60, follow_redirects=True)
    check_reply(reply)
    target.write_bytes(reply.content)


def main() -> None:
    key = (dotenv_values(REPO_ROOT / ".env").get("ROBOFLOW_API_KEY") or "").strip()
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY in .env first.")
    (OUT_DIR / "source").mkdir(parents=True, exist_ok=True)

    found = find_images(key)
    # A thread pool suits this because each call waits on the network. pool.map keeps input order.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        records = list(pool.map(lambda item: fetch_record(item, key), found))
        list(pool.map(fetch_photo, records))
    (OUT_DIR / "records.json").write_text(json.dumps(records, indent=0), encoding="utf-8")
    print(f"Saved {len(records)} photos and their labels to {OUT_DIR}")


if __name__ == "__main__":
    main()
