"""
test_extraction.py — Run Claude extraction on all demo_images and compare against ground truth.

Usage:
  python test_extraction.py              # full run (all 48 images)
  python test_extraction.py --quick      # re-test only cases that failed in last run
  python test_extraction.py --debug 037  # print raw Claude JSON for PU code ending in 037

Results are saved to test_results.json after every run so you can read/diff them.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import anthropic

IMAGES_DIR   = Path(__file__).parent / "demo_images"
MANIFEST     = IMAGES_DIR / "manifest.csv"
RESULTS_FILE = Path(__file__).parent / "test_results.json"

EXTRACTION_PROMPT = """You are extracting data from a Nigerian INEC EC8A election result sheet.
This is Form EC8A — Statement of Result of Poll from Polling Unit, 2023 Presidential Election.

Extract the following and return ONLY valid JSON — no prose, no markdown fences:

{
  "meta": {
    "state": "",
    "lga": "",
    "ward": "",
    "polling_unit": "",
    "pu_code": ""
  },
  "summary": {
    "registered_voters": null,
    "accredited_voters": null,
    "valid_votes_summary": null,
    "valid_votes_table": null,
    "rejected_ballots": null
  },
  "parties": {
    "APC":  { "figures": null, "words": "", "mismatch": false, "reconciled": null },
    "LP":   { "figures": null, "words": "", "mismatch": false, "reconciled": null },
    "PDP":  { "figures": null, "words": "", "mismatch": false, "reconciled": null },
    "NNPP": { "figures": null, "words": "", "mismatch": false, "reconciled": null },
    "other_parties_total": null
  },
  "flags": {
    "quality": "good",
    "stamp_obscures": false,
    "rotation": false,
    "unreadable_fields": [],
    "sum_valid_votes": null,
    "sum_party_votes": null,
    "sum_delta": null
  }
}

Rules:
- Use null for any field you cannot read with confidence.
- For party votes: prefer the figures column. If obscured, use the words column.
- If a party's votes box appears empty or blank (no digits written), record it as 0, not null.
  Note: receiving 0 votes is common — a blank figures cell means 0, not "unknown".
  Use null only when the cell clearly has a number written in it that you cannot decipher.
- Zero-figure override: If a party's figures column is blank/reads as 0 BUT the words column
  clearly states a positive number (e.g. 'two hundred and twenty nine'), the blank cell was
  misidentified as zero — the party actually received votes. In this case, skip Step 2 for this
  party and immediately set reconciled = words-value.
  Example: LP.figures=0, LP.words='two hundred and twenty nine'=229 → LP.reconciled=229.
  Exception: if words = 'NOUGHT', 'NIL', 'ZERO', or any zero-word, both columns agree on 0.
- Words column bleed: the IN WORDS cell is narrow and adjacent to the polling agent signature
  column. Text from outside the cell (names, signatures, stray words) may bleed in. When reading
  the words column, extract only the leading number-words (e.g. "one hundred and twenty three").
  Stop at the first word that is clearly not part of a number (e.g. a person's name, "VOTE",
  "VOTES", "ONLY", a signature). Ignore anything after that point.
  If after stripping bleed text the words column is empty, treat it as unreadable (null words).
- Number-word misspellings: Nigerian EC8A forms are handwritten and often contain non-standard
  spellings. Parse these phonetic variants as their intended numbers:
  "NINTY" or "NINETY" = 90, "FOURTY" or "FORTY" = 40, "EIGHTY" = 80, "SEVENTY" = 70,
  "NOUGHT" or "NAUGHT" = 0, "HUNDERED" or "HUNDERD" = hundred.
  Apply the same principle to any word that clearly sounds like a number despite irregular spelling.

Step 1 — Per-field cross-check (figures vs words) + valid_votes dual check:
- For each party: compare figures vs words column as before (mismatch=true/false).
- For valid_votes: the total appears in TWO places on this form — read both:
    • valid_votes_summary = the figure in numbered row 7 at the top of the form
    • valid_votes_table   = the figure in the "TOTAL VALID VOTES" row at the bottom of the party table
  Reconcile them:
    - If they agree → use that value as the target total.
    - If they disagree → use the one closer to the sum of party votes you can already read.
    - If both are unreadable → skip Step 2 entirely.

Step 2 — Sum-based reconciliation across all parties:
NOTE: Step 2 OVERRIDES the "prefer figures" rule above. Use whichever of figures or words
minimizes |candidate_sum − valid_votes|, even if that means choosing words over figures.
- Use the reconciled valid_votes from Step 1 as the target total.
- For each party with mismatch=true you have two candidate values: figures and the words-parsed number.
  Parties with mismatch=false have one fixed value.
- IMPORTANT: other_parties_total must be read directly from the form by summing the individually
  listed unlisted-party rows. Do NOT back-calculate it as (valid_votes − LP − APC − PDP − NNPP).
  Back-calculating cascades misreads: if LP is wrong, other_parties_total absorbs the error and
  falsely validates the wrong LP value. Read each minor party's figures cell independently.
- Enumerate every combination of figures/words choices across the mismatched parties.
  For each combination compute: candidate_sum = LP + APC + PDP + NNPP + other_parties_total (null = 0).
- Choose the combination whose candidate_sum is closest to the reconciled valid_votes.
  Tiebreak: prefer words over figures. When two combinations give equal candidate_sum distance,
  always choose the one that uses the words-parsed value, not the figures value.
  Examples:
  • Tie: LP_figures=149, LP_words=179, both give candidate_sum=292 → set LP.reconciled=179.
  • Clear sum winner: LP_figures=10, LP_words='NINTY' (=90), valid_votes=95, other parties=5.
    LP=10 gives candidate_sum=15 (delta=80). LP=90 gives candidate_sum=95 (delta=0).
    LP=90 wins — set LP.reconciled=90. The sum proves the figures reading is wrong.
  • Narrow winner despite large absolute delta: LP=343 (fixed), APC_figures=28, APC_words=23,
    valid_votes=252, others=15. APC=28: sum=386, delta=134. APC=23: sum=381, delta=129.
    APC=23 wins (129 < 134) — choose words even though both are far from valid_votes.
- Set each party's reconciled = the value chosen by the winning combination.
  For parties with mismatch=false, reconciled = figures.
- Set sum_valid_votes = the reconciled valid_votes used as target.
  Set sum_party_votes = sum of all reconciled values (treat null as 0).
  Set sum_delta = abs(sum_party_votes - sum_valid_votes), or null if either total is null.

Step 3 — Overflow correction (run if sum_party_votes > sum_valid_votes):
A) Hard overflow (sum_delta > 15): A party's reconciled value is "impossible" if it exceeds
   the remaining budget:
     budget_for_P = sum_valid_votes - sum_of_all_other_parties_reconciled
   If any party P has reconciled > budget_for_P AND budget_for_P >= 0:
   - P's value was likely misread with an extra digit (e.g. '4' recorded as '44').
   - Set P.reconciled = null and add P to unreadable_fields.
   - Update sum_party_votes and sum_delta accordingly.
   Apply this check to each party independently (largest offender first if multiple).

B) Stray-mark removal — MANDATORY: After Step 3A (or if sum_delta ≤ 15), run this check.
   Loop through every party P in this order: APC, LP, PDP, NNPP, other_parties_total.
   For each P, test ALL THREE conditions:
     1. P.words is null or empty string  (no written-out confirmation of this number)
     2. P.reconciled > 0
     3. sum_party_votes - P.reconciled == sum_valid_votes  (removing P hits the target exactly)
   If all three hold for any P: set P.reconciled = 0, update sum_party_votes, update sum_delta, STOP.
   Worked example (must match this logic exactly):
     APC.reconciled=3, APC.words='', LP=71, others=1. sum_party=75, sum_valid=72, sum_delta=3.
     Check APC: words='' ✓, reconciled=3>0 ✓, 75-3=72=sum_valid ✓ → ALL THREE hold.
     → Set APC.reconciled=0. sum_party=72, sum_delta=0. STOP.

- quality = "good" if most fields readable; "poor" if stamp/blur affects some; "unreadable" if mostly unreadable.
- unreadable_fields: list plain-English descriptions of what you could NOT read.
- Return ONLY the JSON object."""


def load_image_b64(path: Path) -> str:
    from PIL import Image
    import base64, io
    img = Image.open(path)
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def extract(client: anthropic.Anthropic, img_b64: str) -> dict:
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text",  "text": EXTRACTION_PROMPT},
            ],
        }],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def classify(ex_lp, ex_apc, gt_lp, gt_apc):
    if gt_lp is None and gt_apc is None:
        return "NULL"
    lp_delta  = (ex_lp  - gt_lp)  if (ex_lp  is not None and gt_lp  is not None) else None
    apc_delta = (ex_apc - gt_apc) if (ex_apc is not None and gt_apc is not None) else None
    lp_match  = (lp_delta  == 0)
    apc_match = (apc_delta == 0)
    if lp_match and apc_match:                   return "EXACT"
    if lp_match or apc_match:                    return "PARTIAL"
    if ex_lp is None and ex_apc is None:         return "NULL"
    return "WRONG"


def load_manifest():
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_previous_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return None


def run_tests(client, rows, debug_suffix=None):
    """
    Run extractions on `rows`. If debug_suffix is set, print raw Claude JSON for
    the matching PU code and exit.
    Returns list of per-image result dicts.
    """
    results = []

    total = len(rows)
    print(f"\nTesting {total} images against ground truth...\n")
    print(f"{'#':<4} {'PU Code':<18} {'GT LP':>6} {'GT APC':>6}  {'Ex LP':>6} {'Ex APC':>6}  {'LP d':>6} {'APC d':>6}  Quality  Status     Words (LP / APC)")
    print("-" * 100)

    for i, row in enumerate(rows, 1):
        pu_code  = row["polling_unit_code"]
        filename = row["filename"]
        gt_lp    = int(float(row["lp_ground_truth"]))  if row["lp_ground_truth"]  else None
        gt_apc   = int(float(row["apc_ground_truth"])) if row["apc_ground_truth"] else None

        img_path = IMAGES_DIR / filename
        if not img_path.exists():
            print(f"{i:<4} {pu_code:<18} {'MISSING IMAGE':>50}")
            results.append({"pu_code": pu_code, "filename": filename,
                            "status": "MISSING", "error": "image not found"})
            continue

        try:
            img_b64  = load_image_b64(img_path)

            # Debug mode: dump raw Claude response for one image and exit
            if debug_suffix and pu_code.endswith(debug_suffix):
                print(f"\n[DEBUG] Raw Claude response for {pu_code} ({filename}):\n")
                raw_result = extract(client, img_b64)
                print(json.dumps(raw_result, indent=2))
                print()
                sys.exit(0)

            result   = extract(client, img_b64)
            parties  = result.get("parties", {})
            flags    = result.get("flags", {})
            quality  = flags.get("quality", "?")

            def _best(p):
                v = p.get("reconciled")
                return v if v is not None else p.get("figures")

            ex_lp        = _best(parties.get("LP",  {}))
            ex_apc       = _best(parties.get("APC", {}))
            lp_words     = parties.get("LP",  {}).get("words", "")
            apc_words    = parties.get("APC", {}).get("words", "")
            lp_mismatch  = parties.get("LP",  {}).get("mismatch", False)
            apc_mismatch = parties.get("APC", {}).get("mismatch", False)
            sum_delta    = flags.get("sum_delta")
            sum_vv       = flags.get("sum_valid_votes")
            sum_pv       = flags.get("sum_party_votes")

        except Exception as e:
            print(f"{i:<4} {pu_code:<18} {'ERROR: ' + str(e)[:60]}")
            results.append({"pu_code": pu_code, "filename": filename,
                            "status": "ERROR", "error": str(e)})
            time.sleep(0.5)
            continue

        lp_delta  = (ex_lp  - gt_lp)  if (ex_lp  is not None and gt_lp  is not None) else None
        apc_delta = (ex_apc - gt_apc) if (ex_apc is not None and gt_apc is not None) else None

        lp_str    = str(ex_lp)  if ex_lp  is not None else "null"
        apc_str   = str(ex_apc) if ex_apc is not None else "null"
        lp_d_str  = (f"{lp_delta:+d}"  if lp_delta  is not None else "-")
        apc_d_str = (f"{apc_delta:+d}" if apc_delta is not None else "-")

        status = classify(ex_lp, ex_apc, gt_lp, gt_apc)

        conflict_tag = ""
        if lp_mismatch or apc_mismatch:
            parts = []
            if lp_mismatch:  parts.append("LP")
            if apc_mismatch: parts.append("APC")
            conflict_tag = f" [CONFLICT:{'/'.join(parts)}]"

        gt_lp_s  = str(gt_lp)  if gt_lp  is not None else "-"
        gt_apc_s = str(gt_apc) if gt_apc is not None else "-"
        words_str = f"{lp_words or '-'} / {apc_words or '-'}"
        sum_str   = (f"sum: {sum_pv} vs valid_votes {sum_vv} (delta={sum_delta})"
                     if sum_delta is not None else f"sum: valid_votes={sum_vv}")

        print(f"{i:<4} {pu_code:<18} {gt_lp_s:>6} {gt_apc_s:>6}  {lp_str:>6} {apc_str:>6}  {lp_d_str:>6} {apc_d_str:>6}  {quality:<8} {status}{conflict_tag}")
        if status in ("WRONG", "PARTIAL") or conflict_tag:
            print(f"     {'':18} {'':>6} {'':>6}  {'':>6} {'':>6}  {'':>6} {'':>6}           words: {words_str}")
            print(f"     {'':18} {'':>6} {'':>6}  {'':>6} {'':>6}  {'':>6} {'':>6}           {sum_str}")

        results.append({
            "pu_code":      pu_code,
            "filename":     filename,
            "status":       status,
            "gt_lp":        gt_lp,
            "gt_apc":       gt_apc,
            "ex_lp":        ex_lp,
            "ex_apc":       ex_apc,
            "lp_delta":     lp_delta,
            "apc_delta":    apc_delta,
            "lp_words":     lp_words,
            "apc_words":    apc_words,
            "lp_mismatch":  lp_mismatch,
            "apc_mismatch": apc_mismatch,
            "quality":      quality,
            "sum_delta":    sum_delta,
            "sum_vv":       sum_vv,
            "sum_pv":       sum_pv,
            "conflict_tag": conflict_tag,
        })

        time.sleep(0.3)

    return results


def print_summary(results, prev_results=None):
    # Build lookup from previous run if available
    prev_by_pu = {}
    if prev_results:
        for r in prev_results.get("results", []):
            prev_by_pu[r["pu_code"]] = r.get("status")

    total   = len(results)
    exact   = sum(1 for r in results if r["status"] == "EXACT")
    partial = sum(1 for r in results if r["status"] == "PARTIAL")
    wrong   = sum(1 for r in results if r["status"] == "WRONG")
    failed  = sum(1 for r in results if r["status"] in ("NULL", "MISSING", "ERROR"))

    print("-" * 100)
    print(f"\nResults ({total} images):")
    print(f"  Exact match (LP + APC both correct): {exact:>3}  ({exact/total*100:.1f}%)")
    print(f"  Partial match (one correct):         {partial:>3}  ({partial/total*100:.1f}%)")
    print(f"  Wrong (extracted but incorrect):     {wrong:>3}  ({wrong/total*100:.1f}%)")
    print(f"  Failed / null:                       {failed:>3}  ({failed/total*100:.1f}%)")

    if prev_by_pu:
        improved = [r for r in results if prev_by_pu.get(r["pu_code"]) != r["status"]
                    and r["status"] == "EXACT" and prev_by_pu.get(r["pu_code"]) != "EXACT"]
        regressed = [r for r in results if prev_by_pu.get(r["pu_code"]) == "EXACT"
                     and r["status"] != "EXACT"]
        if improved:
            print(f"\n  Improvements vs last run: {[r['pu_code'] for r in improved]}")
        if regressed:
            print(f"  Regressions vs last run:  {[r['pu_code'] for r in regressed]}")

    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in errors:
            print(f"  {r['pu_code']}  {r['filename']}: {r['error']}")

    print()


def save_results(results, is_quick=False):
    prev = load_previous_results()

    output = {
        "quick_run": is_quick,
        "results":   results,
        "summary": {
            "total":   len(results),
            "exact":   sum(1 for r in results if r["status"] == "EXACT"),
            "partial": sum(1 for r in results if r["status"] == "PARTIAL"),
            "wrong":   sum(1 for r in results if r["status"] == "WRONG"),
            "failed":  sum(1 for r in results if r["status"] in ("NULL", "MISSING", "ERROR")),
        },
    }

    # For quick runs, merge with full-run results so the file always has all 48
    if is_quick and prev:
        merged_by_pu = {r["pu_code"]: r for r in prev.get("results", [])}
        for r in results:
            merged_by_pu[r["pu_code"]] = r
        merged = list(merged_by_pu.values())
        output["results"] = merged
        output["summary"] = {
            "total":   len(merged),
            "exact":   sum(1 for r in merged if r["status"] == "EXACT"),
            "partial": sum(1 for r in merged if r["status"] == "PARTIAL"),
            "wrong":   sum(1 for r in merged if r["status"] == "WRONG"),
            "failed":  sum(1 for r in merged if r["status"] in ("NULL", "MISSING", "ERROR")),
        }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to {RESULTS_FILE.name}")


def main():
    parser = argparse.ArgumentParser(description="Test Claude EC8A extraction")
    parser.add_argument("--quick",  action="store_true",
                        help="Re-test only cases that failed (PARTIAL/WRONG/NULL) in the last run")
    parser.add_argument("--debug",  metavar="SUFFIX",
                        help="Print raw Claude JSON for the PU code ending with SUFFIX (e.g. 037)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    all_rows = load_manifest()

    if args.debug:
        # Debug mode: run only matching image, dump raw response
        matching = [r for r in all_rows if r["polling_unit_code"].endswith(args.debug)]
        if not matching:
            print(f"No PU code ending with '{args.debug}' found in manifest.")
            sys.exit(1)
        run_tests(client, matching, debug_suffix=args.debug)
        return

    if args.quick:
        prev = load_previous_results()
        if not prev:
            print("No previous results found — running full test instead.")
        else:
            prev_statuses = {r["pu_code"]: r["status"] for r in prev.get("results", [])}
            failing_pu = {pu for pu, s in prev_statuses.items() if s != "EXACT"}
            rows = [r for r in all_rows if r["polling_unit_code"] in failing_pu]
            print(f"[QUICK] Re-testing {len(rows)} non-EXACT cases from last run.")
            results = run_tests(client, rows)
            print_summary(results, prev)
            save_results(results, is_quick=True)
            return

    # Full run
    results = run_tests(client, all_rows)
    print_summary(results)
    save_results(results, is_quick=False)


if __name__ == "__main__":
    main()
