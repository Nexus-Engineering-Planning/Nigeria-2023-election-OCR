"""
ocr_parties.py

Runs EasyOCR on downloaded INEC result sheet images and extracts vote counts
for all 18 parties that contested Nigeria's 2023 presidential election.

Outputs:
    training_data/party_votes.csv       — one row per image, columns for all parties
    training_data/annotations_full.csv  — party_votes merged with existing annotations.csv

Usage:
    py ocr_parties.py [--images DIR] [--annotations CSV] [--output DIR]
"""

import argparse
import csv
import re
import os
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# All 18 parties that contested the 2023 Nigerian presidential election
# Ordered as they typically appear on the EC8A form (alphabetical by acronym)
# ---------------------------------------------------------------------------
PARTIES = [
    "A",        # Action Alliance
    "AA",       # Action Alliance
    "AAC",      # African Action Congress
    "ADC",      # African Democratic Congress
    "ADP",      # Action Democratic Party
    "APC",      # All Progressives Congress
    "APGA",     # All Progressives Grand Alliance
    "APM",      # Allied Peoples Movement
    "APP",      # Action Peoples Party (was incorrectly listed as ARP)
    "BP",       # Boot Party
    "LP",       # Labour Party
    "NNPP",     # New Nigeria Peoples Party
    "NRM",      # National Rescue Movement
    "PDP",      # Peoples Democratic Party
    "PRP",      # Peoples Redemption Party
    "SDP",      # Social Democratic Party
    "YPP",      # Young Progressives Party
    "ZLP",      # Zenith Labour Party
]

# Regex: match a bare number (possibly with commas), standalone
NUMBER_RE = re.compile(r"\d+")

# ---------------------------------------------------------------------------
# Word-to-number conversion for IN WORDS validation
# Handles 0–999 including Nigerian form conventions like "ONE SIXTY THREE" (163)
# ---------------------------------------------------------------------------
_WORD_MAP = {
    "zero":0,"nil":0,"none":0,
    "one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
    "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,
    "twenty":20,"thirty":30,"forty":40,"fifty":50,
    "sixty":60,"seventy":70,"eighty":80,"ninety":90,
    # common misspellings seen on Nigerian ballots
    "fourty":40,"ninty":90,"tweny":20,"eghty":80,"sivty":60,
}

def words_to_number(text):
    """
    Convert a written vote count to int. Returns None if unparseable.

    Handles:
      "ZERO" / "NIL"            → 0
      "THIRTY NINE"             → 39
      "ONE HUNDRED AND THREE"   → 103
      "ONE SIXTY THREE"         → 163  (hundreds without 'hundred' keyword)
    """
    if not text:
        return None
    tokens = re.sub(r"[^a-z ]", " ", text.lower()).split()
    tokens = [t for t in tokens if t not in ("and", "the", "votes", "only")]
    if not tokens:
        return None

    # Standard additive parse with 'hundred' keyword
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

    # Detect "ONE SIXTY THREE" pattern: first token is 1–9, rest add up to 10–99
    # Reinterpret as hundreds-digit × 100 + remainder
    if (len(tokens) >= 2
            and tokens[0] in _WORD_MAP
            and 1 <= _WORD_MAP[tokens[0]] <= 9
            and "hundred" not in tokens
            and any(_WORD_MAP.get(t, 0) >= 10 for t in tokens[1:])):
        remainder = sum(_WORD_MAP.get(t, 0) for t in tokens[1:])
        total = _WORD_MAP[tokens[0]] * 100 + remainder

    return total if total >= 0 else None


def clean_number(text):
    """
    Extract the dominant number from an OCR string.
    Handles noise like '2e', '35L', '1,234', 'I85' (I misread as 1).
    Returns the largest contiguous digit sequence, or '' if none found.
    """
    # Common OCR confusions: I/l -> 1, O -> 0, S -> 5, Z -> 2, B -> 8
    normalized = (text
                  .replace("I", "1").replace("l", "1")
                  .replace("O", "0").replace("o", "0")
                  .replace("S", "5").replace("Z", "2")
                  .replace("B", "8").replace(",", ""))
    matches = NUMBER_RE.findall(normalized)
    if not matches:
        return ""
    # Return the longest numeric match (most likely to be the vote count)
    return max(matches, key=len)


def extract_votes_from_lines(text_lines, img_height=1400, img_width=1000):
    """
    Given a list of (text, bbox, conf) tuples sorted top-to-bottom,
    find each party acronym and grab:
      - the nearest number to its right (IN FIGURES column)
      - the nearest word-group further right (IN WORDS column)

    y_band is relative to image height so it works across varying scan sizes.

    Returns dict: {
        party:             figures value (str),
        party_conf:        EasyOCR confidence for figures read (float),
        party_words:       raw IN WORDS text (str),
    }
    """
    results = {}
    for p in PARTIES:
        results[p]            = ""
        results[f"{p}_conf"]  = ""
        results[f"{p}_words"] = ""

    # Vertical tolerance: 2.5% of image height (~35px on a 1400px image, ~53px on 2157px)
    y_band = img_height * 0.025

    # Words column cutoff: text starting past 65% of image width is likely
    # the NAME/SIGNATURE column — exclude it from IN WORDS capture
    words_col_max_x = img_width * 0.65

    # Build flat word list: (text, x_center, y_center, x0, x1, conf)
    words = []
    for text, bbox, conf in text_lines:
        x0, y0, x1, y1 = bbox
        words.append((text.strip(), (x0 + x1) / 2, (y0 + y1) / 2, x0, x1, conf))

    for party in PARTIES:
        for text, cx, cy, x0, x1, conf in words:
            normalized = re.sub(r"[^A-Z]", "", text.upper())
            if normalized == party:
                # ── IN FIGURES: first number to the right of the acronym ──
                # Store (x_left, x_right, number, confidence) so we can use
                # the right edge of the figures token as the words-zone boundary
                fig_candidates = [
                    (wx0, wx1, clean_number(wtext), wconf)
                    for wtext, wcx, wcy, wx0, wx1, wconf in words
                    if abs(wcy - cy) < y_band and wx0 > x0 and clean_number(wtext)
                ]
                if fig_candidates:
                    fig_candidates.sort(key=lambda c: c[0])
                    results[party]           = fig_candidates[0][2]
                    results[f"{party}_conf"] = round(fig_candidates[0][3], 3)

                # ── IN WORDS: text tokens right of figures, left of signature zone ──
                # Use the right edge (wx1) of the figures token so wide numbers
                # like "135" don't clip the start of the words column
                fig_x_end = fig_candidates[0][1] if fig_candidates else x1 + 10
                word_tokens = [
                    wtext
                    for wtext, wcx, wcy, wx0, wx1, wconf in words
                    if abs(wcy - cy) < y_band
                    and wx0 > fig_x_end
                    and wx0 < words_col_max_x
                    and not re.search(r"\d", wtext)  # exclude anything containing a real digit
                    and len(wtext) > 1               # skip single-char noise
                ]
                if word_tokens:
                    results[f"{party}_words"] = " ".join(word_tokens).upper()

                break

    return results


def ocr_image(image_path, reader):
    """
    Run EasyOCR on a single image.
    Returns:
        lines:      list of (text, bbox, conf) — bbox is (x0, y0, x1, y1)
        img_height: pixel height of the image
        img_width:  pixel width of the image
    """
    import numpy as np
    pil_img = Image.open(image_path).convert("RGB")
    img_width, img_height = pil_img.size
    img_np  = np.array(pil_img)
    results = reader.readtext(img_np)
    # EasyOCR returns: [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence), ...]
    lines = []
    for bbox_pts, text, conf in results:
        xs   = [p[0] for p in bbox_pts]
        ys   = [p[1] for p in bbox_pts]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        lines.append((text, bbox, conf))
    return lines, img_height, img_width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images",      default="training_data/images",      help="Directory of .gif images")
    parser.add_argument("--annotations", default="training_data/annotations.csv", help="Existing annotations CSV")
    parser.add_argument("--output",      default="training_data",              help="Output directory")
    args = parser.parse_args()

    images_dir  = Path(args.images)
    out_dir     = Path(args.output)
    image_files = sorted(images_dir.glob("*.gif"))

    if not image_files:
        print(f"No .gif images found in {images_dir}")
        return

    # Auto-detect GPU — uses CUDA if available, falls back to CPU gracefully.
    # With a GTX 1050 (4 GB VRAM) + 32 GB RAM, EasyOCR fits comfortably on GPU.
    # Set env var FORCE_CPU=1 to override and run on CPU only.
    import easyocr
    try:
        import torch
        use_gpu = torch.cuda.is_available() and not os.environ.get("FORCE_CPU")
    except ImportError:
        use_gpu = False

    device_label = "GPU" if use_gpu else "CPU"
    print(f"Found {len(image_files)} images. Loading EasyOCR ({device_label})...")
    reader = easyocr.Reader(["en"], gpu=use_gpu)
    print(f"EasyOCR loaded on {device_label}. Running OCR...\n")

    # -----------------------------------------------------------------------
    # Checkpointing: load already-processed rows so a crashed run can resume
    # -----------------------------------------------------------------------
    party_votes_path = out_dir / "party_votes.csv"
    fields = ["polling_unit_code", "image_file"]
    for p in PARTIES:
        fields += [p, f"{p}_conf", f"{p}_words"]

    already_done = set()
    party_rows   = []

    if party_votes_path.exists():
        with open(party_votes_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_done.add(row["polling_unit_code"])
                party_rows.append(row)
        print(f"Checkpoint: {len(already_done)} images already processed — skipping.\n")

    # Open CSV in append mode so each row is flushed immediately
    csv_file   = open(party_votes_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
    if not already_done:
        csv_writer.writeheader()   # write header only on a fresh run

    total      = len(image_files)
    remaining  = [p for p in image_files
                  if "/".join(p.stem.split("_")[:4]) not in already_done]
    done_count = len(already_done)

    try:
        for img_path in remaining:
            done_count += 1
            stem  = img_path.stem
            parts = stem.split("_")
            puc   = "/".join(parts[:4])

            print(f"[{done_count}/{total}] {puc} ... ", end="", flush=True)

            try:
                lines, img_h, img_w = ocr_image(img_path, reader)
                votes = extract_votes_from_lines(lines, img_height=img_h, img_width=img_w)
                found    = sum(1 for p in PARTIES if votes.get(p))
                low_conf = [p for p in PARTIES
                            if votes.get(f"{p}_conf") and float(votes[f"{p}_conf"]) < 0.5]
                status = f"{found} parties found"
                if low_conf:
                    status += f" | low-conf: {', '.join(low_conf)}"
                print(status)
            except Exception as e:
                print(f"ERROR: {e}")
                votes = {p: "" for p in PARTIES}
                for p in PARTIES:
                    votes[f"{p}_conf"]  = ""
                    votes[f"{p}_words"] = ""

            row = {"polling_unit_code": puc, "image_file": img_path.name}
            row.update(votes)
            party_rows.append(row)
            csv_writer.writerow(row)
            csv_file.flush()   # write to disk immediately — crash-safe

    finally:
        csv_file.close()

    print(f"\nParty votes saved to {party_votes_path}")

    # -----------------------------------------------------------------------
    # Merge with annotations.csv → annotations_full.csv
    # -----------------------------------------------------------------------
    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        print("annotations.csv not found — skipping merge")
        return

    party_by_puc = {r["polling_unit_code"]: r for r in party_rows}

    merged = []
    ann_fields = []
    with open(annotations_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ann_fields = reader.fieldnames or []
        for row in reader:
            puc = row["polling_unit_code"]
            pv  = party_by_puc.get(puc, {})
            for party in PARTIES:
                row[f"ocr_{party}"]       = pv.get(party, "")
                row[f"ocr_{party}_conf"]  = pv.get(f"{party}_conf", "")
                row[f"ocr_{party}_words"] = pv.get(f"{party}_words", "")
            merged.append(row)

    new_cols = []
    for p in PARTIES:
        new_cols += [f"ocr_{p}", f"ocr_{p}_conf", f"ocr_{p}_words"]
    full_fields = ann_fields + new_cols
    full_path = out_dir / "annotations_full.csv"
    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=full_fields)
        writer.writeheader()
        writer.writerows(merged)

    print(f"Full annotations saved to {full_path}")

    # -----------------------------------------------------------------------
    # Accuracy check 1: OCR figures vs ground-truth labels (4 main parties)
    # -----------------------------------------------------------------------
    print("\n--- Accuracy check: OCR figures vs ground-truth labels ---")
    for party in ("APC", "PDP", "LP", "NNPP"):
        match = total_compared = 0
        for row in merged:
            existing = row.get(party, "").strip()
            ocr_val  = row.get(f"ocr_{party}", "").strip()
            if existing and ocr_val:
                total_compared += 1
                try:
                    if abs(int(existing.split(".")[0]) - int(ocr_val)) <= 1:
                        match += 1
                except ValueError:
                    pass
        if total_compared:
            pct = 100 * match // total_compared
            print(f"  {party:<6} {match}/{total_compared} ({pct}%)")
        else:
            print(f"  {party:<6} no overlapping labels")

    # -----------------------------------------------------------------------
    # Accuracy check 2: figures vs words agreement (all parties, all rows)
    # High agreement = low stamp interference; low agreement = flag for review
    # -----------------------------------------------------------------------
    print("\n--- Accuracy check: IN FIGURES vs IN WORDS agreement ---")
    fig_word_match = fig_word_total = 0
    unparseable = 0
    for row in merged:
        for party in PARTIES:
            fig = row.get(f"ocr_{party}", "").strip()
            wrd = row.get(f"ocr_{party}_words", "").strip()
            if not fig or not wrd:
                continue
            fig_word_total += 1
            word_num = words_to_number(wrd)
            if word_num is None:
                unparseable += 1
                continue
            try:
                if abs(int(fig) - word_num) <= 1:
                    fig_word_match += 1
            except ValueError:
                pass
    if fig_word_total:
        pct = 100 * fig_word_match // fig_word_total
        print(f"  {fig_word_match}/{fig_word_total} rows agree ({pct}%)")
        print(f"  {unparseable} word values could not be parsed")
        print(f"  {fig_word_total - fig_word_match - unparseable} mismatches → likely stamp interference")


if __name__ == "__main__":
    main()
