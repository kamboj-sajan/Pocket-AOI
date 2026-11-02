# Pocket AOI

From one phone photo, find assembly mistakes on a through-hole circuit board by comparing it with a known-good golden board, and say what each mistake will break.

Built for the Meta Global AI Developer Hackathon (manufacturing). The full spec is in [AGENTS.md](AGENTS.md).

**Status:** Block 0, setup. The two model smoke tests below run; the inspection pipeline is not built yet, and there are no results to report.

## Models

| Role | Model |
| --- | --- |
| Board outline and part masks | SAM 3.1 by Meta, loaded from [unsloth/sam3.1](https://huggingface.co/unsloth/sam3.1), a transformers-format conversion of Meta's weights under the SAM License. It is not an official Meta release. |
| Crop verdicts and consequence text | [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) by Meta, Apache 2.0 |

## Setup

Needs Python 3.11 or newer. These commands use [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env    # then set LLM_API_KEY
```

## Smoke tests

```bash
.venv/bin/pytest                                                  # no network or GPU needed
.venv/bin/pocket-aoi smoke-llm                                    # generated test board to Muse Glimmer
.venv/bin/pocket-aoi smoke-llm --image photo.jpg                  # your own photo
.venv/bin/pocket-aoi smoke-sam --image photo.jpg                  # finds "printed circuit board"
.venv/bin/pocket-aoi smoke-sam --image photo.jpg --prompt resistor
```

`smoke-sam` saves its overlay to `runs/smoke/sam_overlay.png`.

## Taking photos

Hold the phone 25–35 cm above and roughly parallel to the board, with the board filling most of the frame. Use diffuse light with no flash, on a plain background that contrasts with the board. Save photos as JPEG; on an iPhone, set Settings > Camera > Formats to Most Compatible.
