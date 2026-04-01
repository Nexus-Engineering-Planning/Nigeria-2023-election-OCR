"""
validate.py

Validation pipeline for INEC EC8A extracted data.

Reads annotations_full.csv (output of ocr_parties.py) and applies two
distinct layers of flags:

  OCR QUALITY FLAGS  — the pipeline is uncertain about a value it extracted.
                       Action: re-check the number, not the document.

  DOCUMENT INTEGRITY FLAGS — the form itself is anomalous regardless of
                             whether the numbers were read correctly.
                             Action: investigate the document, not the OCR.

Outputs:
  training_data/validation_results.csv   — all rows with flag columns added
  training_data/human_review_queue.csv   — only rows where any flag is set,
                                           with a plain-English review_reasons column

Usage:
    py validate.py [--input CSV] [--output DIR]
"""

import argparse
import csv
import re
from pathlib import Path


# ── Parties (must match ocr_parties.py) ──────────────────────────────────────
PARTIES = [
    "A", "AA", "AAC", "ADC", "ADP", "APC", "APGA", "APM", "APP",
    "BP", "LP", "NNPP", "NRM", "PDP", "PRP", "SDP", "YPP", "ZLP",
]

# ── Thresholds ────────────────────────────────────────────────────────────────
CONF_THRESHOLD       = 0.5   # EasyOCR confidence below this → low-confidence flag
ARITHMETIC_TOLERANCE = 5     # allow ±5 votes margin for minor rounding on the form
OVERVOTE_TOLERANCE   = 0     # total votes must not exceed accredited (strict)


# ── Word-to-number (duplicated from ocr_parties.py for standalone use) ────────
_WORD_MAP = {
    "zero":0,"nil":0,"none":0,
    "one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
    "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,
    "twenty":20,"thirty":30,"forty":40,"fifty":50,
    "sixty":60,"seventy":70,"eighty":80,"ninety":90,
    "fourty":40,"ninty":90,"tweny":20,"eghty":80,"sivty":60,
}

def words_to_number(text):
    """Convert a written vote count to int. Returns None if unparseable."""
    if not text:
        return None
    tokens = re.sub(r"[^a-z ]", " ", text.lower()).split()
    tokens = [t for t in tokens if t not in ("and", "the", "votes", "only")]
    if not tokens:
        return None

    total, current = 0, 0
    for t in tokens:
        if t == "hundred":
            current = (current or 1) * 100
        elif t == "thousand":
            total += (current or 1) * 1000
            current = 0
        elif t in _WORD_MAP:
            current += _WORD_MAP[t]
    total += current

    # "ONE SIXTY THREE" style (hundreds without 'hundred' keyword)
    if (len(tokens) >= 2
            and tokens[0] in _WORD_MAP
            and 1 <= _WORD_MAP[tokens[0]] <= 9
            and "hundred" not in tokens
            and any(_WORD_MAP.get(t, 0) >= 10 for t in tokens[1:])):
        remainder = sum(_WORD_MAP.get(t, 0) for t in tokens[1:])
        total = _WORD_MAP[tokens[0]] * 100 + remainder

    return total if total >= 0 else None


# ── Helpers ───────────────────────────────────────────────────────────────────
def to_int(val):
    """Safely convert a CSV value to int. Returns None if blank or unparseable."""
    if val is None:
        return None
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None

def to_float(val):
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None

def is_present(val):
    """Return True if an authentication flag column indicates presence (1 or '1')."""
    v = to_int(val)
    return v == 1


# ── Flag functions ─────────────────────────────────────────────────────────────

def ocr_quality_flags(row):
    """
    Returns (flags_dict, reasons_list) for OCR quality issues.
    These mean: we're not confident in the number — check the read.
    """
    flags   = {}
    reasons = []

    # 1. Figures vs words mismatch ─────────────────────────────────────────────
    mismatched_parties = []
    for p in PARTIES:
        fig  = to_int(row.get(f"ocr_{p}", ""))
        wrd  = row.get(f"ocr_{p}_words", "").strip()
        if fig is None or not wrd:
            continue
        word_num = words_to_number(wrd)
        if word_num is None:
            continue
        if abs(fig - word_num) > 1:
            mismatched_parties.append(f"{p} ({fig}≠{word_num})")

    flags["flag_figures_words_mismatch"] = bool(mismatched_parties)
    if mismatched_parties:
        reasons.append(f"Figures/words mismatch: {', '.join(mismatched_parties)}")

    # 2. Arithmetic check: sum of party votes vs total_use ─────────────────────
    party_sum = 0
    parties_read = 0
    for p in PARTIES:
        v = to_int(row.get(f"ocr_{p}", ""))
        if v is not None:
            party_sum += v
            parties_read += 1

    total_use = to_int(row.get("total_use", ""))
    arith_fail = False
    if total_use is not None and parties_read >= 4:
        # Only flag if we read enough parties to make the check meaningful
        if abs(party_sum - total_use) > ARITHMETIC_TOLERANCE:
            arith_fail = True
            reasons.append(
                f"Arithmetic fail: party sum {party_sum} ≠ total valid votes {total_use} "
                f"(diff={party_sum - total_use:+d})"
            )
    flags["flag_arithmetic_fail"] = arith_fail

    # 3. Low OCR confidence ────────────────────────────────────────────────────
    low_conf_parties = []
    for p in PARTIES:
        conf = to_float(row.get(f"ocr_{p}_conf", ""))
        if conf is not None and conf < CONF_THRESHOLD:
            low_conf_parties.append(f"{p}({conf:.2f})")

    flags["flag_low_confidence"] = bool(low_conf_parties)
    if low_conf_parties:
        reasons.append(f"Low OCR confidence: {', '.join(low_conf_parties)}")

    return flags, reasons


def document_integrity_flags(row):
    """
    Returns (flags_dict, reasons_list) for document integrity issues.
    These mean: the form itself may not be legally valid — investigate.
    """
    flags   = {}
    reasons = []

    # 4. Missing presiding officer name ────────────────────────────────────────
    flags["flag_missing_officer_name"] = not is_present(
        row.get("presiding_officer_name_present")
    )
    if flags["flag_missing_officer_name"]:
        reasons.append("Missing presiding officer name")

    # 5. Missing presiding officer signature ───────────────────────────────────
    flags["flag_missing_officer_signature"] = not is_present(
        row.get("presiding_officer_signature_present")
    )
    if flags["flag_missing_officer_signature"]:
        reasons.append("Missing presiding officer signature")

    # 6. Missing polling agent signature ───────────────────────────────────────
    flags["flag_missing_polling_agent_sig"] = not is_present(
        row.get("polling_agent_signature_present")
    )
    if flags["flag_missing_polling_agent_sig"]:
        reasons.append("Missing polling agent signature")

    # 7. Missing black stamp ───────────────────────────────────────────────────
    # Note: the orange mark on forms is a pre-printed INEC security watermark,
    # NOT an authentication stamp. Only the black stamp is checked here.
    flags["flag_missing_black_stamp"] = not is_present(row.get("black_stamp"))
    if flags["flag_missing_black_stamp"]:
        reasons.append("Missing black stamp")

    # 8. Overvote: total votes cast > accredited voters ────────────────────────
    party_sum    = sum(
        v for p in PARTIES
        if (v := to_int(row.get(f"ocr_{p}", ""))) is not None
    )
    accredited   = to_int(row.get("accredited_num", ""))
    overvote     = (
        accredited is not None
        and accredited > 0
        and party_sum > accredited + OVERVOTE_TOLERANCE
    )
    flags["flag_overvote"] = overvote
    if overvote:
        reasons.append(
            f"Overvote: {party_sum} votes cast but only {accredited} accredited"
        )

    # 9. Zero-vote anomaly: all parties show zero but accredited count > 0 ─────
    all_zero = (
        parties_read := sum(
            1 for p in PARTIES if to_int(row.get(f"ocr_{p}", "")) is not None
        )
    ) >= 4 and party_sum == 0 and (accredited or 0) > 10

    flags["flag_all_zero_votes"] = all_zero
    if all_zero:
        reasons.append(
            f"All votes zero but {accredited} voters were accredited — "
            "possible extraction failure or anomaly"
        )

    return flags, reasons


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",  default="training_data/annotations_full.csv",
        help="Path to annotations_full.csv (output of ocr_parties.py)"
    )
    parser.add_argument(
        "--output", default="training_data",
        help="Directory for validation_results.csv and human_review_queue.csv"
    )
    args   = parser.parse_args()
    in_path  = Path(args.input)
    out_dir  = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run ocr_parties.py first.")
        return

    # ── Read input ────────────────────────────────────────────────────────────
    with open(in_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {in_path}\n")

    # ── All flag column names (for consistent CSV output) ─────────────────────
    ocr_flag_cols = [
        "flag_figures_words_mismatch",
        "flag_arithmetic_fail",
        "flag_low_confidence",
    ]
    integrity_flag_cols = [
        "flag_missing_officer_name",
        "flag_missing_officer_signature",
        "flag_missing_polling_agent_sig",
        "flag_missing_black_stamp",
        "flag_overvote",
        "flag_all_zero_votes",
    ]
    all_flag_cols = ocr_flag_cols + integrity_flag_cols

    # ── Counters for summary ──────────────────────────────────────────────────
    counts   = {col: 0 for col in all_flag_cols}
    reviewed = 0
    results  = []

    for row in rows:
        ocr_flags,  ocr_reasons  = ocr_quality_flags(row)
        int_flags,  int_reasons  = document_integrity_flags(row)

        all_flags   = {**ocr_flags, **int_flags}
        all_reasons = ocr_reasons + int_reasons
        needs_review = any(all_flags.values())

        for col, val in all_flags.items():
            if val:
                counts[col] += 1
        if needs_review:
            reviewed += 1

        out_row = dict(row)
        out_row.update({k: int(v) for k, v in all_flags.items()})
        out_row["flag_count"]      = sum(1 for v in all_flags.values() if v)
        out_row["needs_review"]    = int(needs_review)
        out_row["review_reasons"]  = " | ".join(all_reasons) if all_reasons else ""
        results.append(out_row)

    # ── Write validation_results.csv ──────────────────────────────────────────
    base_fields = list(rows[0].keys()) if rows else []
    extra_fields = all_flag_cols + ["flag_count", "needs_review", "review_reasons"]
    out_fields  = base_fields + [f for f in extra_fields if f not in base_fields]

    results_path = out_dir / "validation_results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Validation results → {results_path}")

    # ── Write human_review_queue.csv ──────────────────────────────────────────
    queue_rows  = [r for r in results if r["needs_review"]]
    # Sort: document integrity flags first (more serious), then OCR quality
    queue_rows.sort(
        key=lambda r: (
            -sum(int(r.get(c, 0)) for c in integrity_flag_cols),
            -sum(int(r.get(c, 0)) for c in ocr_flag_cols),
        )
    )
    queue_path  = out_dir / "human_review_queue.csv"
    queue_fields = [
        "polling_unit_code", "state_name", "lga_name", "ward_name", "unit_name",
        "image_file", "flag_count", "needs_review", "review_reasons",
    ] + all_flag_cols
    with open(queue_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=queue_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(queue_rows)
    print(f"Human review queue  → {queue_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(results)
    print(f"\n{'─'*55}")
    print(f"VALIDATION SUMMARY  ({total} documents)")
    print(f"{'─'*55}")
    print(f"\n  OCR QUALITY FLAGS")
    for col in ocr_flag_cols:
        label = col.replace("flag_", "").replace("_", " ").title()
        pct   = 100 * counts[col] // total if total else 0
        print(f"    {label:<35} {counts[col]:>5}  ({pct}%)")

    print(f"\n  DOCUMENT INTEGRITY FLAGS")
    for col in integrity_flag_cols:
        label = col.replace("flag_", "").replace("_", " ").title()
        pct   = 100 * counts[col] // total if total else 0
        print(f"    {label:<35} {counts[col]:>5}  ({pct}%)")

    print(f"\n  Total needing review: {reviewed}/{total} ({100*reviewed//total if total else 0}%)")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
