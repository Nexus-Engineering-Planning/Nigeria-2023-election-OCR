# Nigeria 2023 Election OCR — Project TODO

## 1. Data Collection
- [x] Download `AllPollingUnitsInfo.csv`, `voter_info.csv`, `stamp_sig_missing.csv`, `LGALevelResult.csv`
- [x] Write `build_training_data.py` — stratified image sampler with annotations
- [ ] Run full 500-image sample (`py build_training_data.py --sample 500`)
- [ ] Expand to 2,000+ images for a production-grade training set
- [ ] Add `LICENSE` file to clarify data reuse rights (e.g. CC BY 4.0)

## 2. Data Preparation
- [ ] Convert downloaded `.gif` images to `.jpg`/`.png` for framework compatibility
- [ ] Split into train / validation / test sets (e.g. 70/15/15)
- [ ] Reserve ~10% as a **fixed hold-out benchmark** — never used in training
- [ ] Write a data dictionary documenting all column names in the CSVs

## 3. Model Training
- [ ] Choose layout detection framework (Surya, PaddleOCR, or docTR)
- [ ] Fine-tune layout model on INEC result sheet format (tables, stamps, signatures, key-value pairs)
- [ ] Fine-tune OCR model on handwritten vote counts from labeled `voter_info.csv` rows
- [ ] Log training run metadata: date, dataset version, hyperparameters, framework version

## 4. Model Drift Protection
- [ ] **Baseline metrics** — record accuracy on the hold-out set immediately after first training run (stamp/sig F1, vote count extraction rate)
- [ ] **Confidence thresholds** — flag any predictions below a set confidence score for human review instead of auto-accepting
- [ ] **Periodic re-evaluation** — re-run the model against the fixed hold-out set on a regular schedule and log results to a `drift_log.csv`
- [ ] **Distribution monitoring** — alert if prediction distributions shift significantly (e.g. sudden spike in "stamp missing" rate, unusual party vote distributions)
- [ ] **Re-training triggers** — define criteria for triggering a new training run (e.g. hold-out accuracy drops >5% from baseline, or >500 new human-reviewed corrections accumulate)
- [ ] **Data versioning** — tag each training data snapshot (e.g. `v1.0-500imgs`) so every model can be traced back to the exact dataset it was trained on
- [ ] **Changelogs** — document any known INEC document format changes and which model version addressed them

## 5. Validation Pipeline
- [ ] Cross-reference extracted vote counts against `LGALevelResult.csv` (LGA-level ground truth)
- [ ] Flag polling units where sum of unit-level votes diverges from LGA totals
- [ ] Validate votes cast <= accredited voters for each unit
- [ ] Cross-check numbers vs. written words where both are present on the form

## 6. Analysis & Output
- [ ] Reproduce anomaly detection: missing stamps/signatures, illegible documents, result discrepancies
- [ ] Generate state- and LGA-level summary reports
- [ ] Map polling units with anomalies using lat/lng coordinates from `AllPollingUnitsInfo.csv`
- [ ] Publish final dataset with SHA-256 checksums for integrity verification
