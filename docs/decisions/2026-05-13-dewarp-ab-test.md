# ADR: Make UVDocNet dewarping opt-in for crowdsource images

**Date:** 2026-05-13  
**Status:** Accepted  
**Deciders:** Femi (StayGIS / E Don Kast)

---

## Context

`ec8a_extract.py` and `demo/server.py` used UVDocNet dewarping (YOLO OBB layout detection + UVDocNet rectification) as the **default** for all `--source-channel crowdsource` images. The reasoning was that citizen phone photos often have perspective distortion and page curl that hurts Claude's OCR accuracy.

Prior to formalising this, no controlled comparison existed between the dewarped and raw paths.

## Decision

**Run dewarping only when explicitly requested.** In `ec8a_extract.py` this means replacing `--no-dewarp` (opt-out) with `--dewarp` (opt-in). In `demo/server.py` the trigger is the request header `x-server-dewarp: true` instead of the previous `x-client-preprocessed: true`. The default crowdsource path now passes raw images directly to Claude.

## Evidence

A/B test on 9 images (2026-05-13):

| Category | Images | Dewarp better | Equivalent | Dewarp worse |
|---|---|---|---|---|
| Control (pre-unwrapped) | 3 | 0 | 1 | 2 |
| Hard (raw consolidated) | 3 | 0 | 2 | 1 |
| Worst-case (Yuxi rejects) | 3 | 0 | 1 | 2 |
| **Total** | **9** | **0** | **4** | **5** |

Key findings:
- `flag_dewarp_failed` was **false for all 9 images** — YOLO orientation succeeded everywhere, so the fallback path was never triggered. The regressions were not due to orientation failures.
- Quality flag **worsened** on 2 images with dewarp (`hard_01_01_01_002`: unreadable vs poor; `worst_22_03_02_008`: poor vs good). It improved on 0.
- Applying dewarp to already-flat (pre-unwrapped) images caused severe extraction errors on `ctrl_01_01_01_002` — wrong state, LGA, codes, and all party figures.
- For the 3 worst-case images, nodewarp produced usable output on 2 of 3 vs dewarp's 1 of 3.
- SSIM values for dewarped images ranged 0.44–0.58, confirming the pipeline makes substantial visual changes that are often net-negative.

Full per-image results: `ab_tests/2026-05-13/REPORT.md` (gitignored — local benchmark artifact).

## Consequences

**What changes:**
- `ec8a_extract.py`: `--no-dewarp` flag removed; `--dewarp` flag added. Default is no dewarping for all channels.
- `demo/server.py`: `x-client-preprocessed` header removed; `x-server-dewarp: true` header added. Default crowdsource path skips UVDocNet entirely.
- All model loading, fallback logic, and `DewarpError` handling remain intact — the infrastructure is preserved for opt-in use.

**What does not change:**
- `scripts/dewarp.py` — unchanged.
- The `dewarp_applied` field in server records — still populated (false by default, true when explicitly requested).
- The public API of `dewarp_to_bytes` / `load_models` / `DewarpError`.

**Risks and mitigations:**
- *Small sample (n=9):* The verdict is strong enough to act on now. If a larger run (50–100 raw images) reverses the finding for severely warped images, dewarp can be promoted back — but as a targeted gate, not a blanket default.
- *Future preprocessing:* The next highest-leverage investment is browser-side preprocessing (jscanify / OpenCV.js) for live corner detection and perspective transform before upload. This addresses the actual quality bottleneck (framing, blur) without the server-side cost of YOLO + UVDocNet on every request.
- *Model maintenance:* UVDocNet weights (`models/rect_model.pkl`) and YOLO weights (`models/rotation_22may.pt`) are still required for `--dewarp` to function. They are not removed from the repo.
