"""
ocr_full_extract.py

Full extraction from INEC EC8A presidential election result sheets.
Extracts everything possible from each image and validates against ground truth.

Outputs:
    training_data/ocr_results.csv       — all extracted fields per image
    training_data/annotations_full.csv  — ocr_results merged with annotations.csv

Usage:
    py ocr_full_extract.py [--images DIR] [--annotations CSV] [--output DIR] [--lookup CSV]
"""

import argparse
import csv
import re
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# Parties that appeared on the 2023 Nigerian presidential election EC8A forms.
# NOTE: This list may need trimming — user flagged that some scraped party data
# includes newly registered parties not present in 2023. Any party acronym
# detected on a form NOT in this list will be flagged via unknown_party_detected.
# ---------------------------------------------------------------------------
PARTIES = [
    "A", "AA", "AAC", "ADC", "ADP", "APC", "APGA", "APM", "ARP",
    "BP", "LP", "NNPP", "NRM", "PDP", "PRP", "SDP", "YPP", "ZLP",
]

NIGERIAN_STATES = {
    "ABIA", "ADAMAWA", "AKWA IBOM", "ANAMBRA", "BAUCHI", "BAYELSA", "BENUE",
    "BORNO", "CROSS RIVER", "DELTA", "EBONYI", "EDO", "EKITI", "ENUGU",
    "FCT", "ABUJA", "GOMBE", "IMO", "JIGAWA", "KADUNA", "KANO", "KATSINA",
    "KEBBI", "KOGI", "KWARA", "LAGOS", "NASARAWA", "NIGER", "OGUN", "ONDO",
    "OSUN", "OYO", "PLATEAU", "RIVERS", "SOKOTO", "TARABA", "YOBE", "ZAMFARA",
}

DEFAULT_LOOKUP = "nigeria_polling_units.csv"

NUMBER_RE = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# Ground truth lookup
# ---------------------------------------------------------------------------

def load_polling_units_lookup(csv_path):
    """
    Load nigeria_polling_units.csv keyed on the `delimitation` column
    (e.g. '01/01/01/005') which matches our polling_unit_code format directly.
    """
    lookup = {}
    path = Path(csv_path)
    if not path.exists():
        print(f"Warning: lookup file not found at {csv_path}")
        return lookup
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            key = row.get("delimitation", "").strip()
            if key:
                lookup[key] = {
                    "gt_state": row.get("state_name", "").strip(),
                    "gt_lga":   row.get("lga_name",   "").strip(),
                    "gt_ward":  row.get("ward_name",   "").strip(),  # = Registration Area
                    "gt_unit":  row.get("unit_name",   "").strip(),
                }
    print(f"Loaded {len(lookup):,} polling units from lookup")
    return lookup


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clean_number(text):
    """Extract dominant digit sequence from noisy OCR text."""
    normalized = (text
                  .replace("I", "1").replace("l", "1")
                  .replace("O", "0").replace("o", "0")
                  .replace("S", "5").replace("Z", "2")
                  .replace("B", "8").replace(",", ""))
    matches = NUMBER_RE.findall(normalized)
    return max(matches, key=len) if matches else ""


def normalize(text):
    """Uppercase, strip non-alphanumeric for comparison."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def build_word_list(text_lines):
    """Convert EasyOCR output to (text, x0, y0, x1, y1, cx, cy) sorted top-to-bottom."""
    words = []
    for text, bbox in text_lines:
        x0, y0, x1, y1 = bbox
        words.append((text.strip(), x0, y0, x1, y1, (x0+x1)/2, (y0+y1)/2))
    words.sort(key=lambda w: (w[2], w[1]))
    return words


def tokens_right_of(words, anchor_x, anchor_y, x_min=None, y_tol=35):
    x_min = x_min if x_min is not None else anchor_x
    return sorted(
        [w for w in words if abs(w[6] - anchor_y) < y_tol and w[1] > x_min],
        key=lambda w: w[1]
    )


def first_number_right(words, anchor_x, anchor_y, y_tol=35):
    candidates = [
        (w[1], clean_number(w[0]))
        for w in tokens_right_of(words, anchor_x, anchor_y, y_tol=y_tol)
        if clean_number(w[0])
    ]
    return candidates[0][1] if candidates else ""


def loc_match(ocr_val, gt_val):
    """Substring match after normalization. Returns 1/0/'' if either value missing."""
    if not ocr_val or not gt_val:
        return ""
    n_ocr = normalize(ocr_val)
    n_gt  = normalize(gt_val)
    return 1 if n_ocr in n_gt or n_gt in n_ocr else 0


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_header(words):
    """Extract form number, election type, year, and INEC logo presence."""
    result = {
        "form_number":     "",
        "election_type":   "",
        "year":            "",
        "inec_logo_present": 0,
    }
    # Use top 40 tokens for header
    top_words = words[:40]
    full_text = " ".join(w[0] for w in top_words).upper()

    # Form number
    m = re.search(r"EC\s*8\s*[A-Z]?", full_text)
    if m:
        result["form_number"] = re.sub(r"\s+", "", m.group())

    # Election type
    for etype in ["PRESIDENTIAL", "GOVERNORSHIP", "SENATORIAL", "HOUSE OF REPRESENTATIVES", "STATE ASSEMBLY"]:
        if etype in full_text:
            result["election_type"] = etype
            break

    # Year
    m = re.search(r"\b(202\d)\b", full_text)
    if m:
        result["year"] = m.group(1)

    # INEC logo: look for commission name or INEC in top 20% of image
    inec_markers = ["INDEPENDENT NATIONAL ELECTORAL", "INEC", "ELECTORAL COMMISSION"]
    for marker in inec_markers:
        if marker in full_text:
            result["inec_logo_present"] = 1
            break

    return result


def extract_location(words):
    """
    Extract State, LGA, Registration Area (ward), and Polling Unit from the form.
    Used as validation signal against ground truth lookup.
    """
    result = {"ocr_state": "", "ocr_lga": "", "ocr_ward": "", "ocr_unit": ""}

    if not words:
        return result

    ys = [w[2] for w in words]
    y_range = max(ys) - min(ys)
    y_min_loc = min(ys) + y_range * 0.15
    y_max_loc = min(ys) + y_range * 0.55
    loc_words = [w for w in words if y_min_loc < w[2] < y_max_loc]

    SKIP = {"CODE", "AREA", "LOCAL", "STATE", "POLLING", "UNIT", "WARD",
            "GOVERNMENT", "REGISTRATION", "ELECTION", "PRESIDENTIAL", "2023",
            "INDEPENDENT", "NATIONAL", "ELECTORAL", "COMMISSION", "FORM", "EC8A",
            "RESULT", "SHEET", "FEDERAL", "CAPITAL", "TERRITORY", "REPUBLIC",
            "NIGERIA", "NIGERIAN", "GUBERNATORIAL", "SENATORIAL"}

    label_map = {
        "ocr_state": ["STATE"],
        "ocr_lga":   ["LOCAL GOVERNMENT", "LGA"],
        "ocr_ward":  ["REGISTRATION AREA", "REGISTRATION"],
        "ocr_unit":  ["POLLING UNIT"],
    }

    for field, anchors in label_map.items():
        for anchor_text in anchors:
            for w in loc_words:
                if anchor_text in w[0].upper():
                    candidates = tokens_right_of(loc_words, w[1], w[6], y_tol=25)
                    for c in candidates:
                        token_words = set(re.sub(r"[^A-Z ]", "", c[0].upper()).split())
                        # Reject if all words are in SKIP, or any SKIP word is present
                        has_skip = bool(token_words & SKIP)
                        if (len(c[0]) > 2
                                and not token_words.issubset(SKIP)
                                and not has_skip
                                and not re.match(r"^\d+$", c[0].strip())):
                            result[field] = c[0].strip()
                            break
                    if result[field]:
                        break
            if result[field]:
                break

    # State fallback: scan full text for known state names (longest match first)
    if not result["ocr_state"]:
        all_text = " ".join(w[0].upper() for w in words)
        for state in sorted(NIGERIAN_STATES, key=len, reverse=True):
            if state in all_text:
                result["ocr_state"] = state.title()
                break

    return result


def extract_voter_stats(words):
    """Extract handwritten voter statistics. Labels are printed; values are handwritten."""
    result = {
        "registered_voters":  "",
        "accredited_voters":  "",
        "ballots_issued":     "",
        "unused_ballots":     "",
        "spoiled_ballots":    "",
        "rejected_ballots":   "",
        "total_valid_votes":  "",
        "total_used_ballots": "",
    }

    label_map = [
        ("registered_voters",  ["VOTERS ON THE REGISTER", "REGISTER"]),
        ("accredited_voters",  ["ACCREDITED VOTERS", "ACCREDIT"]),
        ("ballots_issued",     ["BALLOT PAPERS ISSUED", "PAPERS ISSUED"]),
        ("unused_ballots",     ["UNUSED BALLOT", "UNUSED"]),
        ("spoiled_ballots",    ["SPOILED BALLOT", "SPOILED"]),
        ("rejected_ballots",   ["REJECTED BALLOT", "REJECTED"]),
        ("total_valid_votes",  ["TOTAL VALID VOTES", "TOTAL VALID"]),
        ("total_used_ballots", ["TOTAL NUMBER OF USED", "USED BALLOTS"]),
    ]

    if not words:
        return result

    ys = [w[2] for w in words]
    y_range = max(ys) - min(ys)
    stat_words = [w for w in words
                  if min(ys) + y_range * 0.25 < w[2] < min(ys) + y_range * 0.68]

    for key, anchors in label_map:
        for anchor in anchors:
            for w in stat_words:
                if anchor in w[0].upper():
                    num = first_number_right(stat_words, w[3], w[6], y_tol=40)
                    if num and 1 <= len(num) <= 6:
                        result[key] = num
                    break
            if result[key]:
                break

    return result


def extract_party_votes(words):
    """
    Extract votes in figures and votes in words for all parties.

    Primary strategy: SN-anchored extraction.
      The EC8A form prints serial numbers 1–18 in the leftmost column of the
      party table. SN is typed (not handwritten) so OCR reads it reliably.
      PARTIES[sn-1] gives the party with certainty regardless of whether the
      acronym token OCRs correctly.

    Fallback strategy: acronym scan.
      For any party not resolved via SN, fall back to scanning for the party
      acronym text and looking right for a number.

    Returns: figures dict, words dict, unknown_parties set, sn_anchor_hits int
    """
    figures   = {p: "" for p in PARTIES}
    words_col = {f"{p}_words": "" for p in PARTIES}
    unknown_parties = set()

    # Short tokens that could be mistaken for party names
    NON_PARTY_TOKENS = {
        "IN", "OF", "NO", "TO", "BY", "AT", "OR", "IF", "AS", "AN", "BE",
        "DO", "GO", "HE", "IT", "ME", "MY", "ON", "UP", "US", "WE", "ID",
        "OK", "CA", "MC", "ML", "NL", "OI", "SC", "TE", "TH", "IS", "AM",
        "ALL", "FOR", "THE", "AND", "NOT", "ARE", "HAS", "ITS", "CAN", "MAY",
        "BUT", "HER", "HIM", "HIS", "OUR", "OUT", "OWN", "PUT", "SAY", "SHE",
        "TOO", "USE", "WAS", "WHO", "WIN", "TEE", "NIC", "NIF", "CIH", "APP",
        "GOV", "LGA", "REG", "SEC", "OFF", "AGT", "NUM", "SIG", "SER", "REF",
        "YES", "NIL", "NEW", "OLD", "ONE", "TWO", "SIX", "TEN", "SUM", "NET",
    }

    if not words:
        return figures, words_col, unknown_parties, 0

    ys  = [w[2] for w in words]
    xs1 = [w[3] for w in words]  # x1 (right edge)
    y_min, y_range = min(ys), max(ys) - min(ys)
    img_w_approx   = max(xs1) if xs1 else 1000

    party_band_min = y_min + y_range * 0.55
    party_band_max = y_min + y_range * 0.92
    band_words     = [w for w in words if party_band_min < w[6] < party_band_max]

    # ------------------------------------------------------------------
    # Step 1: SN-anchored extraction (primary)
    # SN tokens: clean integer 1–18, x-position in left 30% of image
    # ------------------------------------------------------------------
    sn_rows = {}   # sn (int) → (x0, cy)
    sn_x_threshold = img_w_approx * 0.30

    for w in band_words:
        num = clean_number(w[0])
        if num and w[1] < sn_x_threshold:
            try:
                sn = int(num)
                if 1 <= sn <= 18 and sn not in sn_rows:
                    sn_rows[sn] = (w[1], w[6])  # (x0, cy)
            except ValueError:
                pass

    for sn, (sn_x0, sn_cy) in sn_rows.items():
        party = PARTIES[sn - 1]

        # All tokens to the right of SN on the same row, sorted left→right
        row_tokens = sorted(
            [rw for rw in band_words if abs(rw[6] - sn_cy) < 30 and rw[1] > sn_x0],
            key=lambda t: t[1]
        )

        # First token after SN is the party acronym — skip it for figure search
        figure_candidates = []
        skip_first = True
        for rt in row_tokens:
            if skip_first:
                skip_first = False
                continue
            num = clean_number(rt[0])
            if num and 1 <= len(num) <= 6:
                figure_candidates.append((rt[1], num))

        if figure_candidates:
            figures[party] = figure_candidates[0][1]

        # Votes in words: first long non-numeric token after SN, skipping acronym
        skip_first = True
        for rt in row_tokens:
            if skip_first:
                skip_first = False
                continue
            if not clean_number(rt[0]) and len(rt[0]) > 2:
                tok_clean = re.sub(r"[^A-Z]", "", rt[0].upper())
                if tok_clean not in PARTIES:
                    words_col[f"{party}_words"] = rt[0].strip()
                    break

    # ------------------------------------------------------------------
    # Step 2: Acronym fallback for any party not resolved via SN
    # ------------------------------------------------------------------
    for w in band_words:
        text       = w[0].strip()
        normalized = re.sub(r"[^A-Z]", "", text.upper())

        if len(normalized) < 2 or normalized in NON_PARTY_TOKENS:
            continue

        # Unknown party detection: all-caps 3–6 char tokens not in PARTIES
        if (normalized not in PARTIES
                and 3 <= len(normalized) <= 6
                and text.isupper()
                and normalized not in NON_PARTY_TOKENS):
            unknown_parties.add(normalized)

        if normalized in PARTIES:
            party = normalized
            # Only use acronym result if SN extraction didn't already fill this slot
            if figures[party]:
                continue

            x0, cy = w[1], w[6]
            right_tokens = sorted(
                [(rw[1], clean_number(rw[0]))
                 for rw in tokens_right_of(band_words, x0, cy, y_tol=35)
                 if clean_number(rw[0])],
                key=lambda t: t[0]
            )
            if right_tokens:
                figures[party] = right_tokens[0][1]

            if not words_col[f"{party}_words"]:
                right_text = sorted(
                    [rw for rw in tokens_right_of(band_words, x0, cy, y_tol=35)
                     if not clean_number(rw[0]) and len(rw[0]) > 2],
                    key=lambda t: t[1]
                )
                for rt in right_text:
                    if re.sub(r"[^A-Z]", "", rt[0].upper()) not in PARTIES:
                        words_col[f"{party}_words"] = rt[0].strip()
                        break

    return figures, words_col, unknown_parties, len(sn_rows)


def detect_black_stamp(image_path):
    """Detect INEC black stamp via dark pixel density in bottom-right quadrant."""
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            w, h = gray.size
            crop   = gray.crop((w // 2, h // 2, w, h))
            pixels = list(crop.getdata())
            ratio  = sum(1 for p in pixels if p < 60) / len(pixels)
            return 1 if ratio > 0.02 else 0
    except Exception:
        return ""


def detect_orange_watermark(image_path):
    """
    Detect orange INEC watermark via RGB color thresholding on form body (middle 60%).
    Orange hue: R > 180, G in [70, 160], B < 80.
    """
    try:
        with Image.open(image_path) as img:
            rgb  = img.convert("RGB")
            w, h = rgb.size
            # Crop middle 60% vertically
            crop   = rgb.crop((0, int(h * 0.2), w, int(h * 0.8)))
            pixels = list(crop.getdata())
            orange = sum(1 for r, g, b in pixels if r > 180 and 70 <= g <= 160 and b < 80)
            ratio  = orange / len(pixels)
            return 1 if ratio > 0.005 else 0
    except Exception:
        return ""


def detect_presiding_officer(words, img_h):
    """Detect presiding officer presence via certification keywords or bottom-band text."""
    full_text = " ".join(w[0].upper() for w in words)
    for kw in ["HEREBY CERTIFY", "PRESIDING OFFICER", "CERTIFY THAT"]:
        if kw in full_text:
            return 1
    bottom = [w for w in words if w[2] > img_h * 0.85 and len(w[0]) > 3]
    return 1 if len(bottom) >= 2 else 0


# ---------------------------------------------------------------------------
# OCR runner
# ---------------------------------------------------------------------------

def ocr_image(image_path, reader):
    import numpy as np
    img_np = np.array(Image.open(image_path).convert("RGB"))
    results = reader.readtext(img_np)
    lines = []
    for bbox_pts, text, conf in results:
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        lines.append((text, (min(xs), min(ys), max(xs), max(ys))))
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images",      default="training_data/images")
    parser.add_argument("--annotations", default="training_data/annotations.csv")
    parser.add_argument("--output",      default="training_data")
    parser.add_argument("--lookup",      default=DEFAULT_LOOKUP,
                        help="Path to nigeria_polling_units.csv")
    args = parser.parse_args()

    images_dir  = Path(args.images)
    out_dir     = Path(args.output)
    image_files = sorted(images_dir.glob("*.gif"))

    if not image_files:
        print(f"No .gif images found in {images_dir}")
        return

    # Load ground truth lookup
    pu_lookup = load_polling_units_lookup(args.lookup)

    # Load annotations CSV
    ann_by_puc = {}
    ann_fields = []
    ann_path   = Path(args.annotations)
    if ann_path.exists():
        with open(ann_path, encoding="utf-8") as f:
            reader_csv = csv.DictReader(f)
            ann_fields = reader_csv.fieldnames or []
            for row in reader_csv:
                ann_by_puc[row["polling_unit_code"]] = row

    print(f"Found {len(image_files)} images. Loading EasyOCR...")
    import easyocr
    reader = easyocr.Reader(["en"], gpu=True)
    print("Model loaded. Running full extraction...\n")

    ocr_fields = (
        ["polling_unit_code", "image_file"]
        + ["form_number", "election_type", "year"]
        + ["gt_state", "gt_lga", "gt_ward", "gt_unit"]               # ground truth
        + ["ocr_state", "ocr_lga", "ocr_ward", "ocr_unit"]            # OCR extracted
        + ["location_state_match", "location_lga_match",
           "location_ward_match", "location_unit_match"]               # validation
        + ["registered_voters", "accredited_voters", "ballots_issued",
           "unused_ballots", "spoiled_ballots", "rejected_ballots",
           "total_valid_votes", "total_used_ballots"]
        + PARTIES
        + [f"{p}_words" for p in PARTIES]
        + ["black_stamp_ocr", "presiding_officer_present",
           "inec_logo_present", "orange_watermark_present",
           "unknown_party_detected", "sn_anchor_hits"]
    )

    all_rows = []
    total    = len(image_files)

    for i, img_path in enumerate(image_files, 1):
        stem  = img_path.stem
        parts = stem.split("_")
        puc   = "/".join(parts[:4])

        print(f"[{i}/{total}] {puc} ... ", end="", flush=True)

        row = {f: "" for f in ocr_fields}
        row["polling_unit_code"] = puc
        row["image_file"]        = img_path.name

        try:
            # Ground truth from lookup
            gt = pu_lookup.get(puc, {})
            row["gt_state"] = gt.get("gt_state", "")
            row["gt_lga"]   = gt.get("gt_lga",   "")
            row["gt_ward"]  = gt.get("gt_ward",   "")
            row["gt_unit"]  = gt.get("gt_unit",   "")

            # OCR
            raw_lines = ocr_image(img_path, reader)
            words     = build_word_list(raw_lines)
            img_h     = Image.open(img_path).size[1]

            # Header (includes inec_logo_present)
            row.update(extract_header(words))

            # Location (OCR)
            loc = extract_location(words)
            row.update(loc)

            # Location validation: OCR vs ground truth
            row["location_state_match"] = loc_match(loc["ocr_state"], row["gt_state"])
            row["location_lga_match"]   = loc_match(loc["ocr_lga"],   row["gt_lga"])
            row["location_ward_match"]  = loc_match(loc["ocr_ward"],  row["gt_ward"])
            row["location_unit_match"]  = loc_match(loc["ocr_unit"],  row["gt_unit"])

            # Voter stats
            row.update(extract_voter_stats(words))

            # Party votes
            figures, words_col, unknown, sn_hits = extract_party_votes(words)
            row.update(figures)
            row.update(words_col)
            row["unknown_party_detected"] = ",".join(sorted(unknown)) if unknown else 0
            row["sn_anchor_hits"]         = sn_hits

            # Authentication
            row["black_stamp_ocr"]          = detect_black_stamp(img_path)
            row["orange_watermark_present"]  = detect_orange_watermark(img_path)
            row["presiding_officer_present"] = detect_presiding_officer(words, img_h)

            parties_found = sum(1 for p in PARTIES if row[p])
            print(f"{parties_found} parties | sn={sn_hits} | stamp={row['black_stamp_ocr']} "
                  f"| watermark={row['orange_watermark_present']} "
                  f"| logo={row['inec_logo_present']} "
                  f"| gt={'ok' if row['gt_state'] else 'missing'}")

        except Exception as e:
            print(f"ERROR: {e}")

        all_rows.append(row)

    # Write ocr_results.csv
    ocr_path = out_dir / "ocr_results.csv"
    with open(ocr_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ocr_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nOCR results saved to {ocr_path}")

    # Merge with annotations.csv → annotations_full.csv
    if ann_fields:
        ocr_by_puc   = {r["polling_unit_code"]: r for r in all_rows}
        extra_fields = [f"ocr_{f}" for f in ocr_fields
                        if f not in ("polling_unit_code", "image_file")]
        merged = []
        for ann_row in ann_by_puc.values():
            puc   = ann_row["polling_unit_code"]
            ocr_r = ocr_by_puc.get(puc, {})
            for f in ocr_fields:
                if f not in ("polling_unit_code", "image_file"):
                    ann_row[f"ocr_{f}"] = ocr_r.get(f, "")
            merged.append(ann_row)

        full_path   = out_dir / "annotations_full.csv"
        full_fields = ann_fields + extra_fields
        with open(full_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=full_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged)
        print(f"Full annotations saved to {full_path}")

    # -----------------------------------------------------------------------
    # Summaries
    # -----------------------------------------------------------------------
    print("\n--- Accuracy spot-check (OCR vs annotations.csv) ---")
    for party in ["APC", "PDP", "LP", "NNPP"]:
        match = compared = 0
        for ocr_r in all_rows:
            csv_val = ann_by_puc.get(ocr_r["polling_unit_code"], {}).get(party, "").strip()
            ocr_val = ocr_r.get(party, "").strip()
            if csv_val and ocr_val:
                compared += 1
                try:
                    if abs(int(float(csv_val)) - int(ocr_val)) <= 1:
                        match += 1
                except ValueError:
                    pass
        if compared:
            print(f"  {party}: {match}/{compared} ({100*match//compared}%)")

    print("\n--- Location validation (OCR vs ground truth lookup) ---")
    for field in ["location_state_match", "location_lga_match",
                  "location_ward_match", "location_unit_match"]:
        vals = [r[field] for r in all_rows if r[field] != ""]
        if vals:
            pct = 100 * sum(int(v) for v in vals) // len(vals)
            print(f"  {field}: {pct}% ({len(vals)} rows compared)")

    print("\n--- Ground truth coverage ---")
    gt_found = sum(1 for r in all_rows if r["gt_state"])
    print(f"  Lookup hit: {gt_found}/{total} images")

    print("\n--- Authentication ---")
    for field in ["black_stamp_ocr", "orange_watermark_present",
                  "inec_logo_present", "presiding_officer_present"]:
        vals = [r[field] for r in all_rows if r[field] != ""]
        if vals:
            present = sum(int(v) if str(v).isdigit() else 0 for v in vals)
            print(f"  {field} present: {present}/{len(vals)} ({100*present//len(vals)}%)")

    unknown_rows = [r for r in all_rows if r["unknown_party_detected"] and r["unknown_party_detected"] != "0"]
    if unknown_rows:
        print(f"\n--- Unknown parties detected: {len(unknown_rows)} images ---")
        from collections import Counter
        all_unknown = []
        for r in unknown_rows:
            all_unknown.extend(r["unknown_party_detected"].split(","))
        for party, count in Counter(all_unknown).most_common(10):
            print(f"  {party}: {count} images")


if __name__ == "__main__":
    main()
