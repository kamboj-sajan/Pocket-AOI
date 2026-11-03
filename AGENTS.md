# Pocket AOI — Build Prompt

Nov 3, 2026 · Sajan Kamboj

## Read first

Submission closes **5 Nov 2026 at 6:30 AM IST**, about 54 hours after this doc was written, so the build plan below is a two-day sprint rather than ten days ([event schedule](https://internshala.com/competitions/meta-global-ai-developer-hackathon-2026/)). Muse Spark 1.3 is served through the Meta Model API, so the pipeline uses **Muse Glimmer 30B** instead: Meta's open-weight vision model, distilled from Muse Spark and released under Apache 2.0 ([Hugging Face blog](https://huggingface.co/blog/muse-glimmer)). Muse Spark Open, its 816B open-weight sibling, is too large to self-host but is served by hosted providers, so section 2 lists it as the stronger optional backend ([report](https://n8nlab.io/news/meta-open-weight-model-strategy)).

Muse Spark itself is also sold on OpenRouter ([Meta Model API page](https://dev.meta.ai/products/meta-model-api/)), but it is not open-weight, so ask the organizers on Discord before relying on it.

What changed from your proposal:

- Muse Glimmer 30B replaces Muse Spark 1.3 for crop verdicts and consequence text; SAM 3.1 stays.
- Classical vision does more of the deciding. Each golden part is checked at its own spot in the aligned test photo, and a rotation test flags reversed parts, so Glimmer only confirms or rejects suspects.
- The schematic becomes a netlist once (KiCad export or a hand-typed JSON). Consequences are reasoned from that text plus an exact pin-swap table.
- The KiCad pick-and-place stretch goal moves to after the deadline.
- Test boards are dry-fitted rather than soldered, so a defect takes seconds to plant and undo.

How to use this doc:

1. Request access to [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) on Hugging Face now; approval can take time.
2. Export this doc as Markdown and save it in your repo root as `AGENTS.md`.
3. Start each work block by pasting its kickoff message from section 9.

## Verified during Block 0 (3 Nov 2026)

Checked against the live model repos and the installed packages. Where these differ from a later section, these win.

- **SAM weights:** tested with granted access, `Sam3Model.from_pretrained("facebook/sam3.1")` fails with `OSError: ... does not appear to have a file named pytorch_model.bin or model.safetensors`. That repo holds a raw 3.5 GB `sam3.1_multiplex.pt`, and its config names `Sam3VideoModel`. `unsloth/sam3.1` is ungated and in `Sam3Model` format (`model.safetensors`, 1.68 GB, BF16), so it is the default in `configs/default.yaml`. It is a conversion of Meta's weights, not an official Meta release; the README must say so and acknowledge the SAM Materials.
- **Reasoning effort:** Glimmer's chat template reads `Reasoning strength: <value>` from the system prompt, and it silently defaults to `high` when that line is absent. `llm.py` therefore writes the line into every system prompt, which works on any backend using this template, so no backend-specific parameter is needed. On OpenRouter, one identical call each at low and high used 190 and 373 reasoning tokens, so the line appears to be honoured there.
- **OpenRouter:** `meta/muse-glimmer-30b` exists with image input, a 131,072 context, $0.30 per million input tokens and $1.20 per million output tokens, and it accepts `response_format`, `structured_outputs`, `reasoning`, `top_k` and `top_p`. A live Prompt A call on a drawn test composite was accepted with the real `Verdict` schema and took 11–14 s, using about 950 input and 300 output tokens, or roughly $0.0006 per verdict.
- **Cache key:** section 3 keys the cache on image bytes, prompt version and model id. Prompt D has no image, so `llm.py` also hashes the rendered messages; otherwise every consequence card would collide.
- **Extra modules:** `schemas.py` holds the section 4 models and `config.py` loads the YAML and `.env`, since the section 4 layout names no home for either.
- **torchvision:** `Sam3Processor` fails to load without it, so it is in `pyproject.toml` although the section 2 list omits it.
- **Post-processing memory:** `post_process_instance_segmentation` upsamples every kept instance at once as 8-byte integers, and about 100 masks of a 1920 × 2925 photo crashed with a 4.52 GiB buffer error. `segment.py` calls it in chunks and keeps masks as 1-byte bools.
- **SAM without CUDA:** section 2 says SAM 3.1 needs a CUDA GPU, but it also runs on an 8 GB M2 through MPS in float16: 11 s to load and 4–8 s per prompt on a 1920 × 2925 photo. No T4 has been timed yet.
- **First look at part prompts** (one 1986 industrial board, not hand-counted, so not a result): DIP ICs were found with tight masks at 0.67–0.77. The IC prompt also masked the whole board at 0.57, so the Stage 2 area filter is required. Large resistors scored up to 0.66 as "electrolytic capacitor", so per-class thresholds and the class-merge rule need calibrating in Block 1.

## 1. Context and goal

Build **Pocket AOI**: from one phone photo, find assembly mistakes on a through-hole circuit board by comparing it with a known-good golden board, and say what each mistake will break.

- **Who I am:** an ECE (IoT) student. I know C++ well and am still learning Python, so write readable, typed Python and add a short comment wherever an idiom differs from C++.
- **Why it matters:** in student labs, assembly mistakes cost more debugging time than design errors: a reversed electrolytic, an IC seated backwards, a missing resistor. Factory AOI machines cost tens of thousands of dollars; students, makers and small shops inspect by eye.
- **Event:** Meta Global AI Developer Hackathon, theme Push the Frontier (manufacturing). Submission deadline 5 Nov 2026, 6:30 AM IST.
- **Inputs:** a photo of the golden board (once per design), a photo of the board under test, and the board's netlist.
- **Outputs:** an annotated overlay on the test photo, a safe-to-power banner, and one card per fault, for example "C7 missing: U2 loses its decoupling cap, expect random resets."

Done means:

1. `pocket-aoi inspect` runs end to end on a real photo pair and reports its time per stage.
2. It flags missing, reversed, shifted and extra parts on the evaluation boards.
3. Every confirmed fault names its reference designator and a consequence grounded in the netlist.
4. A results table reports detection rate and false positives per board on real planted defects.
5. A Gradio page runs upload, overlay and fault cards.
6. A README and a 2–3 minute demo video are ready to submit.

Out of scope, listed as limitations in the README: solder-joint defects (they sit on the underside of through-hole boards), wrong values in identical bodies (1 kΩ vs 10 kΩ), and fine-pitch SMD boards.

## 2. Hard constraints and models

Use only Meta's open-weight models: **SAM 3.1** for segmentation and **Muse Glimmer 30B** for visual verdicts and consequence text. Do not call the Meta Model API or Muse Spark, and do not train or fine-tune anything.

| Role | Model and weights | Notes |
| --- | --- | --- |
| Board outline and part masks | SAM 3.1: [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) (gated, SAM License), or [unsloth/sam3.1](https://huggingface.co/unsloth/sam3.1), a transformers-format conversion of the same weights (not an official Meta release) | 0.8B parameters, 1.68 GB in BF16. Text prompt in; masks, boxes and scores out. |
| Crop verdicts, silkscreen reading, consequences | Muse Glimmer 30B: [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), Apache 2.0 | 2B Perception Encoder plus 28B text decoder; images and text in, text out; 131K context. |
| Fallback only if Glimmer cannot run | [Llama 3.2 11B Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct), open weights | Weaker, but the one-image composite design suits it. |

SAM 3.1 API shape (check it against the installed transformers version before relying on it):

```python
from transformers import Sam3Model, Sam3Processor
model = Sam3Model.from_pretrained(SAM_ID, dtype=torch.float16).to("cuda").eval()  # T4: float16, not bfloat16
processor = Sam3Processor.from_pretrained(SAM_ID)
inputs = processor(images=img, text="resistor", return_tensors="pt").to("cuda", torch.float16)
with torch.inference_mode():
    out = model(**inputs)
res = processor.post_process_instance_segmentation(
    out, threshold=0.4, mask_threshold=0.5,
    target_sizes=inputs["original_sizes"].tolist())[0]  # res["masks"], res["boxes"], res["scores"]
```

Run Muse Glimmer behind one OpenAI-compatible client set by `.env` (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`), so switching backends is a config change:

1. **Hosted open weights, fastest start:** [OpenRouter meta/muse-glimmer-30b](https://openrouter.ai/meta/muse-glimmer-30b), also served by Together, Fireworks and DeepInfra. Listed at $0.30–0.35 per million input tokens and $1.20–1.50 per million output tokens ([price list](https://cloudprice.net/models/meta-muse-glimmer-30b)); my estimate for the whole evaluation is under one US dollar.
2. **Free on Kaggle:** 2× T4 GPUs, about 30 free GPU hours a week. Serve `unsloth/Muse-Glimmer-30B-GGUF` at `UD-Q4_K_XL` (about 17 GB) plus its `mmproj` vision file with `llama-server -ngl 99 --tensor-split 1,1`, and load SAM 3.1 on GPU 0 ([Unsloth guide](https://unsloth.ai/docs/models/muse-glimmer)).
3. **Local:** 18 GB or more of combined RAM and VRAM runs the 4-bit build; the 2-bit build needs 12–14 GB at some cost in quality.

**Stronger hosted option:** Muse Spark Open, the open-weight 816B mixture-of-experts sibling of Muse Spark (42B parameters active per token), is too big to self-host but is served by OpenRouter, Together, Fireworks, Groq and others ([report](https://n8nlab.io/news/meta-open-weight-model-strategy)). Switching is a change of `LLM_MODEL`; first confirm your provider accepts images for it and check the price. If budget allows, run the evaluation once with it as a 30B-versus-816B ablation.

Model settings:

- Start from Meta's sampling defaults: temperature 1.0, top_p 0.95, top_k 64.
- Reasoning effort: low for verdicts and silkscreen reading, medium for consequences. The parameter name differs by backend (`reasoning_strength` in the transformers chat template); check before use.
- Request JSON-schema output through `response_format` where the backend supports it, and always validate with pydantic.

Environment: Python 3.11, PyTorch with CUDA, latest transformers, opencv-python, numpy, scipy, scikit-image, pycocotools, pydantic v2, gradio, pyyaml, python-dotenv, openai, pytest. SAM 3.1 needs a CUDA GPU; a free Colab or Kaggle T4 is enough.

## 3. Pipeline

Every decision is anchored on the golden board: each golden part is checked at its own location in the aligned test photo, and Muse Glimmer only confirms or rejects the suspects that classical vision raises.

### Stage 0: board setup (once per design, saved in `boards/<board_id>/`)

1. Netlist: for a KiCad design run `kicad-cli sch export netlist --format kicadxml -o netlist.xml board.kicad_sch` and convert it to `netlist.json`. For a kit with a PDF schematic I type `netlist.json` by hand; Prompt C can draft it if time allows.
2. Run Stages 1 and 2 on the golden photo to get the canonical image and part masks.
3. Assign reference designators. Prompt B reads the silkscreen beside each part, and I confirm or fix them in `refs.yaml` or the UI table. Reject a ref that is missing from the netlist or whose prefix does not fit the class (R, C, D, LED, Q, U or IC).
4. Save `golden_profile.json`: canonical size and, per part, ref, class, mask (RLE), box, centroid, package and a polarized flag.

### Stage 1: alignment into one canonical frame

1. Load with EXIF orientation applied; keep the full-resolution image for crops.
2. Prompt SAM 3.1 with "printed circuit board", take the largest mask, fit 4 corners with `cv2.approxPolyDP`, order them and warp to a canonical rectangle (long side 2000 px, the golden board's aspect ratio).
3. Refine test-to-golden with SIFT inside the board mask on CLAHE grayscale (Lowe ratio 0.75, RANSAC at 3 px). Keep the refinement only with 50 or more inliers and a median reprojection error of 2 px or less.
4. Quality gate: board-mask IoU after warping must be at least 0.97. Otherwise return a retake message naming the likely cause (board cut off, steep angle, glare) and stop.
5. For comparison only, normalise lighting: CLAHE on the L channel of LAB plus gray-world white balance.
6. Tall parts lean when the camera is not straight above, so every distance threshold is relative to part size.

### Stage 2: segmentation

1. Class prompts come from the board config. Default: "electrolytic capacitor", "ceramic capacitor", "resistor", "diode", "LED", "integrated circuit", "transistor". Add "IC socket", "potentiometer" or "connector" per kit.
2. Run each prompt on the full canonical image and on 2×2 tiles with 20% overlap. Map tile masks back and merge duplicates (mask IoU above 0.5, keep the higher score).
3. When two prompts hit one object, keep the more specific class if the scores are within 0.1.
4. Keep masks inside the board, of plausible area, scoring at or above a per-class threshold (start at 0.4). Calibrate on golden photos so every real part is found.

### Stage 3: comparison (golden-anchored, tuned for recall)

For each golden part:

1. Match test masks with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). Cost = (1 − IoU) + 0.5 × centroid distance ÷ part diagonal + 1.0 if the class differs; pairs with IoU below 0.1 never match.
2. Crop golden and test at the part's box plus 25% margin, and compute SSIM on the normalised grayscale whatever SAM found.
3. Decide:
   - MISSING: no match and SSIM below the missing threshold.
   - SHIFTED: matched, but the centroid moved more than 30% of the part's shorter side, or IoU is below 0.5.
   - REVERSED or ROTATED (polarized classes only): compare the test crop with the golden crop rotated 0°, 90°, 180° and 270° using SSIM plus edge-map correlation. If a non-zero angle beats 0° by the configured margin, flag it; 180° is REVERSED. A DIP IC can only be reversed.
   - WRONG_PART: matched in place, but the class differs or the appearance is far off.
4. Test masks with no golden match become EXTRA.
5. Every non-OK result becomes a Suspect carrying both crops and the evidence numbers.

### Stage 4: verdicts (Muse Glimmer, Prompt A)

1. Build one composite PNG per suspect: golden crop left, test crop right, same scale, each upscaled so the part spans at least 384 px, labelled GOLDEN and TEST inside the image.
2. Send Prompt A with the part metadata and CV evidence, and parse a `Verdict`.
3. Fusion rule: Glimmer never creates a fault that Stage 3 did not raise.
   - OK with confidence 0.7 or higher: drop the suspect.
   - OK with lower confidence, or UNSURE: keep as needs review.
   - Any fault verdict: confirmed fault, using Glimmer's fault type when its confidence is 0.6 or higher, else Stage 3's.
4. Log both votes for the ablation, and cache every call by SHA-256 of the image bytes, prompt version and model id.

### Stage 5: consequences (netlist facts, rules, Prompt D)

1. Facts per fault: the part's pins and nets, the other parts on each net (ref, value, pin name), named power nets and the supply voltage.
2. Exact pin-swap table for orientation faults. A DIP-N reversed by 180° puts chip pin k on pad ((k − 1 + N/2) mod N) + 1, so a DIP-8 maps 1→5, 4→8, 5→1 and 8→4. A reversed TO-92 swaps pins 1 and 3. A reversed two-pin part swaps its + and − (or A and K) nets.
3. Severity floors set by rules: a reversed electrolytic or IC is CRITICAL (do not power); a reversed diode is at least MAJOR. The model may raise severity but never go below the floor.
4. Prompt D writes the card. Check that the headline starts with the ref and that every cited ref and net exists; retry once with the error, then fall back to a template sentence.

### Stage 6: report

1. Draw the overlay on the original test photo through the inverse homography: red for CRITICAL, orange for MAJOR, yellow for MINOR or needs review, each labelled like "C7 MISSING".
2. Show "Safe to power: NO" when any confirmed fault is CRITICAL.
3. Write `report.json`, `report.html` (one card per fault with both crops, headline, consequence, fix and check-first) and `overlay.png`, sorted by severity, then ref.
4. Save debug images for every stage in `runs/<run_id>/debug/`; they double as demo footage.

## 4. Data contracts and repo layout

Every stage reads and writes pydantic models saved as JSON, so any stage can be rerun or tested alone.

```python
PartClass = Literal["electrolytic_capacitor", "ceramic_capacitor", "resistor", "diode",
                    "led", "ic", "transistor", "other"]
FaultType = Literal["MISSING", "REVERSED", "ROTATED", "SHIFTED", "WRONG_PART", "EXTRA"]

class Part(BaseModel):
    ref: str | None                     # "C7"; None until confirmed
    cls: PartClass
    bbox: tuple[int, int, int, int]     # x0, y0, x1, y1 in canonical pixels
    centroid: tuple[float, float]
    mask_rle: dict                      # COCO RLE via pycocotools
    score: float
    package: str | None                 # "DIP-8", "TO-92", "radial"
    polarized: bool

class GoldenProfile(BaseModel):
    board_id: str
    canonical_size: tuple[int, int]
    parts: list[Part]

class Suspect(BaseModel):
    golden_ref: str | None              # None for EXTRA
    cv_finding: FaultType
    bbox: tuple[int, int, int, int]
    evidence: dict[str, float]          # ssim, iou, offset_ratio, best_rotation_deg, rotation_margin
    composite_path: str

class Verdict(BaseModel):               # Prompt A output
    golden_observation: str
    test_observation: str
    verdict: Literal["OK", "MISSING", "REVERSED", "ROTATED", "SHIFTED",
                     "WRONG_PART", "EXTRA", "UNSURE"]
    confidence: float = Field(ge=0, le=1)
    evidence: str

class Consequence(BaseModel):           # Prompt D output
    headline: str
    consequence: str
    severity: Literal["CRITICAL", "MAJOR", "MINOR"]
    safe_to_power: bool
    fix: str
    check_first: str
    facts_used: list[str]
    uncertain: str

class Fault(BaseModel):
    ref: str | None
    fault_type: FaultType
    status: Literal["confirmed", "needs_review"]
    suspect: Suspect
    verdict: Verdict | None
    consequence: Consequence | None

class Report(BaseModel):
    board_id: str
    test_image: str
    safe_to_power: bool
    faults: list[Fault]
    timings_s: dict[str, float]
    versions: dict[str, str]            # model ids, prompt versions, config hash
```

`netlist.json`, one per board (KiCad export or typed by hand):

```json
{
  "board_id": "ne555-flasher",
  "description": "555 astable LED flasher",
  "supply": {"net": "VCC", "volts": 9},
  "components": [
    {"ref": "U1", "value": "NE555", "type": "ic", "package": "DIP-8",
     "pins": {"1": "GND", "2": "TRIG", "3": "OUT", "4": "RESET",
              "5": "CTRL", "6": "THR", "7": "DIS", "8": "VCC"}},
    {"ref": "C1", "value": "100uF", "type": "electrolytic_capacitor",
     "package": "radial", "pins": {"1": "+", "2": "-"}}
  ],
  "nets": [
    {"name": "VCC", "nodes": ["U1.8", "U1.4", "C1.1"]},
    {"name": "GND", "nodes": ["U1.1", "C1.2"]}
  ]
}
```

Ground truth, one file per evaluation photo:

```json
{"photo": "data/eval/555/555_c03_s1.jpg", "board_id": "ne555-flasher",
 "defects": [{"ref": "C1", "type": "REVERSED"}, {"ref": "R2", "type": "MISSING"}]}
```

Repo layout:

```text
pocket-aoi/
  AGENTS.md                this prompt
  pyproject.toml  .env.example  README.md
  configs/default.yaml     thresholds, class prompts, polarity hints, model ids
  boards/<board_id>/       board.yaml, netlist.json, refs.yaml, golden.jpg, golden_profile.json
  prompts/                 verdict_v1.md, silkscreen_v1.md, netlist_v1.md, consequence_v1.md
  src/pocket_aoi/
    io.py                  load with EXIF, resize, save
    align.py               board quad, homography, refinement, quality gate
    segment.py             SAM 3.1 wrapper, tiling, merging
    compare.py             matching, SSIM, rotation test, suspects
    composite.py           side-by-side crops for the VLM
    llm.py                 OpenAI-compatible client, JSON validation, cache
    netlist.py             KiCad parser, neighbour facts, pin-swap table
    verdict.py             Stage 4
    consequence.py         Stage 5
    report.py              overlay, report.json, report.html
    pipeline.py  cli.py    setup-board, inspect, eval
  app/gradio_app.py
  eval/run_eval.py  eval/metrics.py  eval/ground_truth/
  tests/
  data/  runs/             gitignored
```

CLI:

- `pocket-aoi setup-board --board-id ne555-flasher --golden golden.jpg --netlist netlist.json`
- `pocket-aoi inspect --board-id ne555-flasher --photo test.jpg [--cv-only]`
- `pocket-aoi eval --gt eval/ground_truth/ [--cv-only]`

## 5. Engineering rules

Ship a working end-to-end path first, then improve whichever stage the evaluation shows is weakest.

1. A crude `inspect` must produce a report on a real photo pair by the end of Block 2; polish comes after.
2. Verify APIs instead of guessing: check the installed transformers version and the model cards before writing SAM 3.1 or Glimmer code, and print shapes on the first run.
3. Model calls live only in `segment.py` and `llm.py`.
4. Prompts live in `prompts/*.md` with a version suffix, and the cache key includes the version.
5. Model output is always pydantic-validated. Retry once with the validation error appended, then record UNSURE; never crash a run.
6. Thresholds live in `configs/default.yaml`, and each report records the config hash.
7. Every stage writes debug images to `runs/<run_id>/debug/` (aligned blend, masks, suspect composites).
8. Tests cover synthetic homography recovery (corner error at most 2 px), the pin-swap table, matching and schema round-trips.
9. Before real photos exist, fake defects on golden photos: rotate a part crop 180° and paste it back (REVERSED), or `cv2.inpaint` a part away (MISSING). Report results on real photos only.
10. Write readable Python: type hints, small functions, docstrings, and a one-line comment where Python differs from C++ (comprehensions, generators, context managers, numpy broadcasting).
11. Secrets live in `.env`; commit `.env.example` only and never print keys.
12. Ask me before adding a dependency over about 200 MB or switching frameworks.
13. At the end of each block, report what works, the numbers, what is broken and the next step, and keep the README results table current.

## 6. Build plan: 54 hours to submission

Aim to submit by **Wed 4 Nov, 11 PM IST**, and treat the hours before the 6:30 AM deadline as buffer. All times are IST.

1. **Tonight, 30 minutes, Block 0:** request SAM 3.1 access, create the repo with this file, pick the Glimmer backend and get one test image answered. Collect 2–3 through-hole kits or lab boards you can get today.
2. **Tue 3 Nov, 09:00–13:00, Block 1, segment and align.** Done when SAM finds at least 90% of parts per class on the golden photos (against your hand count) and the synthetic alignment test passes.
3. **Tue 3 Nov, 13:00–20:00, Block 2, compare.** Done when every planted defect in your first real photos shows up as a suspect; extra suspects are fine at this stage.
4. **Tue 3 Nov, evening, your hands-on part:** dry-fit the test boards, plant defects, shoot the evaluation photos and write the ground-truth files (section 7).
5. **Wed 4 Nov, 08:00–13:00, Block 3, verdicts, consequences and report.** Done when `inspect` produces a correct `report.html` for two photos.
6. **Wed 4 Nov, 13:00–18:00, Block 4, UI and evaluation.** Done when the README holds the results tables and the ablation.
7. **Wed 4 Nov, 18:00–23:00, Block 5, ship.** Demo video, README and the submission form.

If you fall behind, cut in this order: Prompt B (type refs by hand), the Gradio setup tab, the ablation, the third kit. Never cut the end-to-end run or the measured results table.

After submission (judging runs 5–9 Nov): KiCad pick-and-place import so no golden board is needed, Prompt C schematic drafting, and SMD boards.

## 7. Evaluation

Measure on real photos only: about 20 planted defects across 2–3 kits, plus clean boards to count false alarms.

- **Boards:** 2–3 cheap through-hole kits with polarized parts, such as a 555 LED flasher, an LM386 amplifier or a 7805 power supply. One correctly assembled copy per kit is the golden board.
- **Fast defect planting:** dry-fit the second copy (leads through the holes, bent slightly, no solder) and seat the IC in a socket, so a defect takes seconds to plant and undo. With only one copy per kit, re-seat the board and move the phone between shots, and state in the README that board-to-board variation was not tested.
- **Defect mix, about 7 per kit:** 2 missing resistors, 1 missing ceramic capacitor, 2 reversed electrolytics, 1 reversed IC, 1 reversed diode or LED, 1 lifted or shifted part.
- **Configurations:** 1–3 defects each, about 10 in total, plus 2 clean configurations per kit.
- **Photos:** 2 shots per configuration with a small change of angle or lighting, roughly 30–40 photos in all.
- **Capture guide, also shown in the UI:** phone 25–35 cm above and roughly parallel to the board; board fills most of the frame; diffuse light, no flash; a plain background that contrasts with the board.
- **Ground truth:** one JSON per photo (format in section 4), written while shooting. Name files `<kit>_<config>_<shot>.jpg`.

Metrics, computed by `eval/metrics.py`:

1. Detection rate: planted defects found at the right ref with the right type, divided by planted defects. REVERSED and ROTATED count as one orientation type. Also report a type-agnostic rate (flagged at the right part, any type).
2. False positives per photo: confirmed faults not in the ground truth, reported separately for clean photos.
3. Needs-review count per photo.
4. Consequence quality: I grade each true fault's card as correct, partly correct or wrong against the schematic.
5. Runtime per photo, by stage.
6. Ablation: CV only versus CV plus Muse Glimmer (false positives removed, recall lost).

The README gets one table per kit and per defect type, plus every miss and false positive with its crop pair.

## 8. Prompts inside the pipeline

Save each prompt as `prompts/<name>_v1.md`. Substitute only the named `{placeholders}` with a small render function (not `str.format`, because the JSON examples contain literal braces), format numbers to two decimals in code, and parse each reply into its schema from section 4.

### Prompt A: crop verdict (Muse Glimmer, image plus text, reasoning effort low)

System:

```text
You are an automated optical inspection (AOI) verifier for through-hole circuit board assemblies.
You receive one image. The left half is the GOLDEN board (known good); the right half is the TEST board.
Both halves show the same location after alignment, at the same scale.
Judge only what is visible. Glare, white balance, blur, shadows and small misalignment (under 10% of the crop) are not defects.
Do not guess component values from colour bands or printing.
Answer with one JSON object and nothing else.
```

User:

```text
Part: {ref} ({part_class}, package {package}, value {value})
Polarity or pin-1 marker to check: {polarity_hint}
Where that marker sits in the GOLDEN half: {golden_marker_hint}
Computer-vision pre-check (may be wrong): {cv_finding}; similarity {ssim}; best rotation {best_rotation_deg} degrees; offset {offset_ratio} of part size

Compare TEST with GOLDEN at this location and choose one verdict:
- OK: same part, same orientation, same position
- MISSING: no part in TEST (empty holes, pads or bare board)
- REVERSED: same part turned 180 degrees (stripe, band, notch or dot on the opposite side)
- ROTATED: multi-pin part turned 90 or 270 degrees
- SHIFTED: part present but displaced, tilted or lifted
- WRONG_PART: a different kind of part, or a visibly different body
- EXTRA: a part in TEST where GOLDEN has none
- UNSURE: the image does not allow a decision

First describe where the marker is in GOLDEN, then in TEST, then decide.
Return exactly:
{"golden_observation": "...", "test_observation": "...", "verdict": "...", "confidence": 0.0, "evidence": "one sentence naming the visible feature"}
```

Polarity hints for `configs/default.yaml`:

| Class | `polarity_hint` |
| --- | --- |
| Electrolytic capacitor | light stripe with minus signs on the can sleeve marks the negative lead |
| Diode | painted band at one end marks the cathode |
| LED | flat edge on the rim marks the cathode |
| IC | notch or dot at the pin-1 end |
| TO-92 transistor | flat printed face |

`golden_marker_hint` comes from setup, for example "stripe toward the left edge of the crop"; leave it empty if unknown.

### Prompt B: silkscreen reference (golden setup, image plus text, reasoning effort low)

System:

```text
You read reference designators printed on PCB silkscreen. Answer with one JSON object and nothing else.
```

User:

```text
The image shows one component outlined in red and the board around it.
Read the reference designator printed for the outlined part, for example R3, C12, U1, D2, Q1 or LED1.
Text may be rotated, small or partly hidden by the part. If several labels are visible, pick the one closest to the outline.
Detected class: {part_class}. Valid references on this board: {candidate_refs}.
If you cannot read it, return null for ref. Never guess.
Return exactly:
{"ref": "R3", "alternatives": [], "confidence": 0.0}
```

Code check: accept only a ref listed in `candidate_refs` whose prefix fits the class; otherwise leave the row for me to fill.

### Prompt C: schematic to netlist draft (optional, kits without a KiCad file, reasoning effort medium)

System:

```text
You transcribe circuit schematics into netlists. Transcribe only what is drawn. Answer with one JSON object and nothing else.
```

User:

```text
Convert this schematic image into the netlist JSON format shown below.
Rules:
- One entry per component with ref, value, type, and package if printed.
- Power symbols (VCC, +9V, GND) become named nets; unnamed nets get N1, N2 and so on.
- Use pin numbers where printed; otherwise pin names (+, -, A, K, E, B, C).
- Pin functions you add from datasheet knowledge (for example NE555 pin 8 = VCC) go in "assumptions".
- Connections you cannot read clearly go in "uncertain".
Format:
{netlist_schema_example}
```

I check every net against the schematic before the draft is used.

### Prompt D: consequence card (Muse Glimmer, text only, reasoning effort medium)

System:

```text
You are a senior electronics engineer explaining an assembly fault to a student before they power the board.
Use only the facts given: netlist, pin-swap table and board description.
If the facts do not support a claim, put it under "uncertain".
Name specific parts, pins and nets. Never invent values or connections.
Answer with one JSON object and nothing else.
```

User:

```text
Board: {board_description}; supply {supply_volts} V on net {supply_net}
Fault: {ref} {fault_type} (verdict confidence {confidence})
Part: {ref} = {value} {part_type}, package {package}
Pins and nets: {pin_net_table}
Neighbours on those nets: {neighbour_table}
Pin-swap table (computed, exact; orientation faults only): {pin_swap_table}
Severity floor set by rules: {severity_floor}

Severity guide:
- CRITICAL: powering can damage parts or the supply (reversed electrolytic or IC, a short across the supply). Do not power.
- MAJOR: the circuit, or one block of it, will not work.
- MINOR: it works, with less margin, more noise or a cosmetic issue.

Return exactly:
{"headline": "at most 12 words, starting with the ref, e.g. C7 missing: U2 loses its decoupling cap",
 "consequence": "2-3 sentences: what happens electrically and what the user will observe",
 "severity": "CRITICAL | MAJOR | MINOR, never below the floor",
 "safe_to_power": true,
 "fix": "one imperative sentence",
 "check_first": "what to measure or inspect if the board was already powered",
 "facts_used": ["U2.8 on net VCC"],
 "uncertain": ""}
```

Format of `{pin_swap_table}`, one line per chip pin. Example for a reversed NE555 whose RESET is tied to VCC:

```text
U1 NE555 reversed (DIP-8). Chip pin -> pad it now sits on -> net of that pad
pin 1 GND   -> pad 5 -> N_CTRL
pin 4 RESET -> pad 8 -> VCC
pin 5 CTRL  -> pad 1 -> GND
pin 8 VCC   -> pad 4 -> VCC
```

## 9. Kickoff messages

Paste one message at the start of each block; each assumes this file sits in the repo root as `AGENTS.md`.

**Block 0, setup (tonight):**

```text
Read AGENTS.md fully. Create the repo skeleton from section 4: pyproject.toml, .env.example, configs/default.yaml, the prompts from section 8 as files, and empty modules with docstrings. Implement llm.py with the OpenAI-compatible client from section 2 and a smoke test that sends one image to Muse Glimmer and prints the reply. Implement segment.py so it loads SAM 3.1 and segments one image with one text prompt. Build nothing else yet. Tell me the commands to run and the output to expect.
```

**Block 1, segment and align:**

```text
Block 1 of AGENTS.md section 6. My photos are in data/raw/<kit>/ (golden.jpg plus test shots).
1. Run SAM 3.1 on each golden photo with the class prompts from configs/default.yaml, on the full image and 2x2 tiles. Save overlays to runs/spike/, print per-class counts so I can compare them with my hand count, and suggest per-class thresholds.
2. Implement align.py exactly as section 3, Stage 1 describes, including the quality gate, plus tests/test_align.py (a synthetic homography recovered within 2 px).
3. Add `pocket-aoi align --golden ... --photo ...`, which saves the aligned image and a checkerboard blend.
Stop and show me the overlays, the counts and the test results.
```

**Block 2, compare:**

```text
Block 2. Implement Stage 0 setup-board (refs from boards/<id>/refs.yaml for now), Stage 3 in compare.py, and composite.py. Test matching and the rotation test with synthetic defects (a golden crop rotated 180 degrees, a part removed with cv2.inpaint). Add `pocket-aoi inspect --cv-only`, which writes suspects.json and the composites. Run it on my real defect photos with eval/ground_truth/ and print, per photo, which planted defects appear as suspects and how many extra suspects there are. Tune thresholds for recall first.
```

**Block 3, verdicts, consequences and report:**

```text
Block 3. Implement Stage 4 (Prompt A, the fusion rule, the cache); netlist.py (KiCad XML parser, neighbour facts, pin-swap table with unit tests: DIP-8 maps 1->5, 4->8, 5->1, 8->4, and TO-92 swaps 1 and 3); Stage 5 (rule floors, Prompt D, ref and net validation); and Stage 6 (overlay on the original photo, report.json, report.html). Run inspect end to end on two photos and show me report.html and the raw model replies.
```

**Block 4, UI and evaluation:**

```text
Block 4. Build app/gradio_app.py with two tabs. "Set up board": golden photo and netlist upload, plus an editable ref table prefilled by Prompt B. "Inspect": photo upload, then the overlay, the safe-to-power banner and fault cards with GOLDEN and TEST crops. Launch with share=True. Then implement eval/run_eval.py and eval/metrics.py per section 7, run the full set and the CV-only ablation, and write the results tables into README.md with every miss and false positive listed.
```

**Block 5, ship:**

```text
Block 5. Freeze features and fix only bugs that break the demo. Finish README.md: the problem, the pipeline, the models used (SAM 3.1 and Muse Glimmer 30B, both Meta open weights), results, limitations and how to run it. Prepare a scripted demo run I can screen-record in under 90 seconds, and give me a submission checklist.
```

Demo video, 2–3 minutes:

1. 0:00–0:20: the lab problem and what factory AOI costs.
2. 0:20–1:30: a live run with golden setup, a test photo, the overlay and one CRITICAL card read aloud.
3. 1:30–2:15: the results table, with one miss and one false positive shown honestly.
4. 2:15–2:40: the open models used, limitations and the next step (KiCad pick-and-place).

## Sources

- [Meta Global AI Developer Hackathon 2026: dates and rules (Internshala)](https://internshala.com/competitions/meta-global-ai-developer-hackathon-2026/)
- [Meta Model API overview](https://dev.meta.ai/products/meta-model-api/)
- [Muse Glimmer release post (Hugging Face)](https://huggingface.co/blog/muse-glimmer)
- [Muse Glimmer: how to run locally (Unsloth)](https://unsloth.ai/docs/models/muse-glimmer)
- [Muse Glimmer 30B provider prices (CloudPrice)](https://cloudprice.net/models/meta-muse-glimmer-30b)
- [SAM 3.1 in transformers format (unsloth/sam3.1)](https://huggingface.co/unsloth/sam3.1)
- [Muse Spark Open release coverage (n8n Lab)](https://n8nlab.io/news/meta-open-weight-model-strategy)
