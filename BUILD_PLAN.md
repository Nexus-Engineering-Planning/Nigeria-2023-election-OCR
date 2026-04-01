# Nigeria 2023 Election OCR — Full Build Plan

**Project:** Automated extraction and validation of vote counts from ~167,000 INEC EC8A polling unit result sheets
**Organisation:** Center for Collaborative Investigative Journalism (CCIJ)
**Stack:** Python · PyTorch · EasyOCR / TrOCR · YOLO · Surya/docTR
**Last updated:** March 2026

---

## Overview

The pipeline has six sequential phases. Each phase feeds into the next, so the order matters. The goal is a fully reproducible system that can process all ~167,000 IReV documents and output validated vote counts, authentication flags, and anomaly reports.

```
Raw GIFs → Pre-processing → Orientation Correction → Layout Detection → OCR → Validation → Analysis
```

Current state: Phase 1 is ~50% done (500 images downloaded, no conversion or splits yet). Phases 2–6 are not started.

---

## Phase 0: Environment Setup

**Goal:** Reproducible Python environment across machines.

**Tasks:**
1. Create `requirements.txt` with pinned versions for all dependencies: `torch`, `torchvision`, `easyocr`, `Pillow`, `pandas`, `numpy`, `scikit-learn`, `ultralytics` (YOLO), `surya-ocr` or `python-doctr`, `boto3` (for Textract).
2. Create `setup.py` or `pyproject.toml` to make the repo installable.
3. Add a `.env.example` for AWS credentials (Textract), DocumentCloud API key.
4. Document GPU requirements (minimum: CUDA-capable GPU with 8GB VRAM; recommend 16GB+ for layout model training).
5. Add `LICENSE` file (CC BY 4.0 recommended for data outputs).

**Deliverable:** `pip install -r requirements.txt` sets up a working environment from scratch.

---

## Phase 1: Data Preparation

**Goal:** Clean, split, and documented training dataset ready for model training.

### 1.1 GIF → PNG Conversion
- Write `convert_images.py` to batch-convert all `.gif` files in `training_data/images/` to `.png` using Pillow.
- Handle multi-frame GIFs (take only the first frame — the result sheets are static scans).
- Preserve original filenames; add `_original_gifs/` backup folder.
- Output: `training_data/images/*.png` (500 files).

### 1.2 Dataset Expansion
- Run `build_training_data.py --sample 2000` to expand to ~2,000 images.
- Confirm stratification is working correctly across all 36 states + FCT.
- Target: at minimum 30 images per state to avoid regional bias.
- Download in batches to avoid rate-limiting from DocumentCloud.

### 1.3 Train / Val / Test Split
- Write `split_dataset.py` with stratified split by state: **70% train / 15% val / 15% test**.
- Reserve a fixed **10% hold-out benchmark** drawn from the test set — this set is never used during any training run and serves only for drift measurement.
- Output: `training_data/splits/train.csv`, `val.csv`, `test.csv`, `holdout.csv`.
- Record random seed in a config file so splits are always reproducible.

### 1.4 Data Dictionary
- Write `DATA_DICTIONARY.md` documenting every column in `annotations.csv`, `voter_info.csv`, `AllPollingUnitsInfo.csv`, `stamp_sig_missing.csv`, and `LGALevelResult.csv`.
- Include: column name, data type, description, example value, null rate.

**Deliverable:** 2,000+ annotated PNG images with documented, reproducible splits.

---

## Phase 2: Document Orientation & Pre-processing

**Goal:** All documents correctly oriented and background-cropped before any OCR runs.

### 2.1 YOLO Logo Detector (Fine-tune)
- The README describes fine-tuning a YOLO model to detect the INEC logo on each EC8A form.
- Fine-tune `YOLOv8` (via `ultralytics`) on a manually labelled subset of ~200 images with INEC logo bounding boxes.
- Use logo position (top-left, top-right, bottom-left, bottom-right) to determine rotation needed.
- Write `correct_orientation.py` to apply rotation using the logo detector.

### 2.2 Background Cropping & Deskew
- Use layout detection output to crop inconsistent white/grey borders.
- Apply deskewing (e.g., `deskew` library or OpenCV Hough line transform) to fix tilted scans.
- Flag documents with dimension ≤ 192×256 px as too small / illegible (matches README finding of 10,000+ illegible papers).

### 2.3 Wrong Document Detection
- Flag documents that are not presidential EC8A forms (e.g., collation papers, EC40G forms, cancelled sheets).
- Heuristic: check for INEC logo presence + expected table structure; if absent, flag for human review.

**Deliverable:** `preprocessing/` module with orientation correction, deskew, and size-based flagging.

---

## Phase 3: Layout Detection Model

**Goal:** For every document, detect and localise: tables, key-value pairs, paragraphs, stamps, signatures, and boxes.

### 3.1 Framework Choice
After evaluating options from the TODO:

| Framework | Pros | Cons |
|---|---|---|
| **Surya** | Purpose-built for document layout, open weights, fast | Less community support, newer |
| **docTR** | PyTorch-native, good API, well-maintained | Primarily text detection, not full layout |
| **PaddleOCR** | Mature, strong layout model | PaddlePaddle dependency (non-PyTorch) |

**Recommendation: Surya** — it outputs layout + reading order + OCR in one pipeline and has open weights that can be fine-tuned on PyTorch. Fall back to docTR if Surya fine-tuning proves too complex.

### 3.2 Training Data for Layout Model
- Manually annotate ~300–500 images with bounding boxes for: `table`, `key_value`, `paragraph`, `stamp`, `signature`, `box`.
- Use [Label Studio](https://labelstud.io/) (free, open-source) for annotation — export in COCO format.
- Aim for at least 50 examples per class, with special attention to stamps and signatures (rare but critical).

### 3.3 Fine-tuning
- Load Surya (or docTR) pre-trained weights.
- Fine-tune on annotated INEC EC8A images using a standard object detection loss (CIoU + classification cross-entropy).
- Validation metric: mAP@0.5 per class. Target: >0.80 mAP on stamps and signatures (highest priority for anomaly detection).
- Log all runs: date, dataset version, hyperparameters, framework version → `training_logs/layout/`.

### 3.4 Anomaly Flagging Rules
Implement the criteria from the README:
- More than 250 bounding boxes → flag
- More than 10 columns → flag
- Fewer than 2 key-value pairs → flag
- Fewer than 1 paragraph → flag
- Fewer than 2 tables → flag
- Missing stamps in expected positions → flag

### 3.5 Authentication Element Detection
Use positional analysis (not classification) to identify:
- Presiding officer signature: bottom portion of page
- Polling agent signatures: inside specific table cells
- Black stamp: bottom-left region
- Orange stamp: middle of page

**Deliverable:** `layout/` module. Output per document: structured JSON with bounding boxes, element counts, authentication flags.

---

## Phase 4: OCR Model

**Goal:** Extract vote counts (numeric, handwritten) and key metadata fields from each document.

### 4.1 Architecture Decision
The two-step approach from the README:
1. Layout model crops individual cells/boxes.
2. OCR model reads text from cropped regions.

For handwritten digit recognition, the best open PyTorch option is **TrOCR** (Microsoft's Transformer-based OCR, available via HuggingFace `transformers`). It handles handwritten text well and can be fine-tuned on domain-specific samples.

For printed text (party names, headers), EasyOCR (already in use) is sufficient — keep it.

### 4.2 Fine-tuning TrOCR
- Crop individual vote count cells from the 2,000 training images using layout model output.
- Each crop should contain one handwritten number.
- Ground truth labels come from `voter_info.csv` (APC, PDP, LP, NNPP vote counts per unit).
- Fine-tune `microsoft/trocr-base-handwritten` on these crops.
- Validation metric: Character Error Rate (CER) and exact-match accuracy on vote count cells. Target: CER < 5%.
- Apply post-processing character substitutions: `I→1`, `O→0`, `S→5`, `Z→2`, `B→8` (already implemented in `ocr_parties.py`).

### 4.3 Full Extraction Pipeline
Combine layout + OCR into `ocr_full_extract.py` (already partially written):
- Header info: form number, election type, INEC logo presence
- Location: state, LGA, ward, polling unit (match against `nigeria_polling_units.csv`)
- All 18 party vote counts + written word representations
- Total valid votes, accredited voters
- Authentication flags (stamp, signatures)

### 4.4 Amazon Textract Integration
The README describes using Textract for large-scale runs (~$500 for all 167k documents). For the training/dev phase, use TrOCR locally. Reserve Textract for production scale-out.
- Implement `textract_extract.py` as a drop-in replacement for local TrOCR when running at scale.
- Use Textract table extraction (`AnalyzeDocument` with `TABLES` feature) for complex table cells.

**Deliverable:** `ocr/` module. Output per document: structured JSON with all extracted fields.

---

## Phase 5: Validation Pipeline

**Goal:** Verify OCR accuracy and flag irregularities, reproducing the three-method approach from the README.

### 5.1 Three-Method Vote Validation
Implement the tiered validation from the README:

- **Method 1:** Match figures vs. written words for APC, PDP, NNPP, LP. Check sum ± 30 (margin for minor parties).
- **Method 2:** Where words ≠ figures, generate all plausible values and pick the combination closest to the stated total (within 10 or 0.1× total).
- **Method 3:** Trust figures if their sum is within ~15 votes of the stated total.

Documents passing none of the three methods are flagged for human review.

### 5.2 LGA-Level Cross-Validation
- Aggregate unit-level OCR results to LGA level.
- Compare against `LGALevelResult.csv` (official FOIA-obtained LGA totals).
- Flag any LGA where aggregated OCR total diverges from official total by >1%.

### 5.3 Accreditation Checks
- Flag polling units where total votes cast > accredited voters (from `voter_info.csv`).
- Flag units where accredited voters = 0 but votes are recorded.

### 5.4 Crowdsource Integration
- For documents flagged by validation, output a review queue CSV compatible with crowdsourcing tools (e.g., Ushahidi, custom web form).
- Accept corrections back as a CSV and merge into final dataset.

**Deliverable:** `validation/` module. Output: `results_validated.csv` with confidence scores and flag columns.

---

## Phase 6: Model Drift Protection

**Goal:** Ensure model quality doesn't degrade over time or across document batches.

**Tasks:**
1. After first successful training run, record baseline metrics on the hold-out set: stamp/sig F1, vote count CER, exact-match rate → `drift_log.csv`.
2. Implement confidence thresholds: any prediction below threshold → route to human review queue instead of auto-accepting.
3. Set re-training triggers: hold-out accuracy drops >5% from baseline, or >500 new human-reviewed corrections accumulate.
4. Tag each training data snapshot (e.g., `v1.0-500imgs`, `v2.0-2000imgs`) using git tags.
5. Document any known INEC format changes in `CHANGELOG.md`.

---

## Phase 7: Scale to Full Dataset & Publication

**Goal:** Run the complete pipeline on all ~167k documents and publish findings.

**Tasks:**
1. Run orientation correction + layout + OCR on all documents in `AllPollingUnitsInfo.csv`.
2. Process in batches of 1,000 with checkpointing (save progress to avoid restarting from scratch on failure).
3. Generate state- and LGA-level summary reports.
4. Map polling units with anomalies using lat/lng from `AllPollingUnitsInfo.csv` (use `geopandas` or export to GeoJSON for mapping tools).
5. Publish final dataset with SHA-256 checksums for each file.
6. Add `LICENSE` (CC BY 4.0) and full data provenance documentation.

---

## Immediate Next Steps (Start Here)

In order of priority:

| # | Task | Script | Effort |
|---|---|---|---|
| 1 | GIF → PNG conversion | `convert_images.py` | 1 hour |
| 2 | Expand to 2,000 images | `build_training_data.py --sample 2000` | 2–3 hours (download time) |
| 3 | Train/val/test split | `split_dataset.py` | 2 hours |
| 4 | `requirements.txt` | manual | 30 min |
| 5 | Label 300 images for layout model | Label Studio | 2–3 days |
| 6 | Fine-tune YOLO for orientation | `train_orientation.py` | 1–2 days |
| 7 | Fine-tune Surya layout model | `train_layout.py` | 3–5 days |
| 8 | Fine-tune TrOCR for digits | `train_ocr.py` | 3–5 days |
| 9 | Build validation pipeline | `validation/` | 2–3 days |
| 10 | Scale to full 167k docs | batch runner | 1 week |

---

## Key Technical Decisions Still Open

1. **Surya vs. docTR for layout** — Surya is recommended but evaluate both on 10 sample documents before committing.
2. **TrOCR vs. PaddleOCR for handwriting** — TrOCR preferred (PyTorch-native, HuggingFace ecosystem), but benchmark against EasyOCR on 50 samples first.
3. **Local TrOCR vs. Amazon Textract at scale** — use local for dev/training, Textract for final 167k run (estimated ~$500–$1,000 total cost).
4. **Annotation tool** — Label Studio (free, self-hosted) recommended over CVAT or Roboflow for this use case.
