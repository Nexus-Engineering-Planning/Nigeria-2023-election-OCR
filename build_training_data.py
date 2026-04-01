"""
build_training_data.py

Downloads a stratified sample of INEC election result sheet images from
DocumentCloud and pairs them with labels from the existing CSVs.

Output structure:
    training_data/
        images/          # downloaded .gif page images
        annotations.csv  # image filename + all available labels
        failed.csv       # records that failed to download

Usage:
    py build_training_data.py [--sample N] [--output DIR] [--delay SECONDS]

Defaults: N=500, DIR=training_data, DELAY=0.5s between requests
"""

import argparse
import csv
import os
import time
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLLING_UNITS_CSV = "AllPollingUnitsInfo.csv"
VOTER_INFO_CSV    = "voter_info.csv"
STAMP_SIG_CSV     = "stamp_sig_missing.csv"

IMAGE_URL_TEMPLATE = (
    "https://assets.documentcloud.org/documents/{doc_id}/pages/{slug}-p1-large.gif"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv_as_dict(path, key_col):
    """Load a CSV into a dict keyed by key_col."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = row[key_col].strip()
            if k:
                rows[k] = row
    return rows


def stratified_sample(rows, group_col, n):
    """
    Sample up to n rows, spread evenly across unique values of group_col.
    rows: list of dicts
    """
    from collections import defaultdict
    import random

    groups = defaultdict(list)
    for r in rows:
        groups[r[group_col]].append(r)

    per_group = max(1, n // len(groups))
    sampled = []
    for group_rows in groups.values():
        sampled.extend(random.sample(group_rows, min(per_group, len(group_rows))))

    # Top up to n if rounding left us short
    remaining = [r for r in rows if r not in sampled]
    random.shuffle(remaining)
    sampled.extend(remaining[: max(0, n - len(sampled))])

    return sampled[:n]


def doc_id_and_slug_from_row(row):
    """
    Extract integer doc_id and slug from the AllPollingUnitsInfo row.
    id column:  '24807553.0'
    URL column: 'https://www.documentcloud.org/documents/24807553-01_01_01_001_crop'
    slug        = everything after the first '-' in the URL path's last segment
    """
    raw_id = row.get("id", "").strip()
    url    = row.get("URL", "").strip()

    if not raw_id or not url:
        return None, None

    try:
        doc_id = str(int(float(raw_id)))
    except ValueError:
        return None, None

    # slug is the part after the numeric id in the URL segment
    last_segment = url.rstrip("/").split("/")[-1]          # e.g. '24807553-01_01_01_001_crop'
    slug = last_segment[len(doc_id) + 1:]                   # e.g. '01_01_01_001_crop'

    if not slug:
        return None, None

    return doc_id, slug


def download_image(doc_id, slug, dest_path, delay):
    """Download page-1 large image from DocumentCloud. Returns True on success."""
    url = IMAGE_URL_TEMPLATE.format(doc_id=doc_id, slug=slug)
    try:
        time.sleep(delay)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 500:          # suspiciously small — probably an error page
            return False
        with open(dest_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample",  type=int,   default=500,           help="Number of images to download (default: 500)")
    parser.add_argument("--output",  type=str,   default="training_data", help="Output directory (default: training_data)")
    parser.add_argument("--delay",   type=float, default=0.5,           help="Seconds between requests (default: 0.5)")
    parser.add_argument("--seed",    type=int,   default=42,            help="Random seed for reproducibility")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    out_dir    = Path(args.output)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load CSVs
    # -----------------------------------------------------------------------
    print("Loading CSVs...")
    voter_info  = load_csv_as_dict(VOTER_INFO_CSV,  "polling_unit_code")
    stamp_sig   = load_csv_as_dict(STAMP_SIG_CSV,   "polling_unit_code")

    # Candidate rows: must have a URL and be legible
    candidates = []
    with open(POLLING_UNITS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] == "exist and not blur" and row["URL"].strip():
                candidates.append(row)

    print(f"  {len(candidates):,} usable polling units found")

    # -----------------------------------------------------------------------
    # Stratified sample by state
    # -----------------------------------------------------------------------
    sample = stratified_sample(candidates, "state_name", args.sample)
    print(f"  Sampled {len(sample):,} units across {len(set(r['state_name'] for r in sample))} states")

    # -----------------------------------------------------------------------
    # Download images + build annotation rows
    # -----------------------------------------------------------------------
    annotation_rows = []
    failed_rows     = []

    annotation_fields = [
        # identifiers
        "polling_unit_code", "state_name", "lga_name", "ward_name", "unit_name",
        "doc_id", "slug", "image_file",
        # voter_info labels
        "registered_num", "accredited_num", "accredit_mark",
        "APC", "PDP", "LP", "NNPP", "total_use",
        # stamp/sig labels
        "presiding_officer_name_present", "presiding_officer_signature_present",
        "polling_agent_signature_present", "black_stamp",
    ]

    total = len(sample)
    for i, row in enumerate(sample, 1):
        puc     = row["polling_unit_code"].strip()
        doc_id, slug = doc_id_and_slug_from_row(row)

        if not doc_id:
            failed_rows.append({"polling_unit_code": puc, "reason": "missing id/slug"})
            continue

        image_file = f"{puc.replace('/', '_')}_{doc_id}.gif"
        dest_path  = images_dir / image_file

        print(f"[{i}/{total}] {puc} ... ", end="", flush=True)

        if dest_path.exists():
            print("cached")
        else:
            ok = download_image(doc_id, slug, dest_path, args.delay)
            if not ok:
                print("FAILED")
                failed_rows.append({"polling_unit_code": puc, "reason": "download error"})
                continue
            print("ok")

        vi  = voter_info.get(puc, {})
        ss  = stamp_sig.get(puc, {})

        annotation_rows.append({
            "polling_unit_code":                  puc,
            "state_name":                         row["state_name"],
            "lga_name":                           row["lga_name"],
            "ward_name":                          row["ward_name"],
            "unit_name":                          row["unit_name"],
            "doc_id":                             doc_id,
            "slug":                               slug,
            "image_file":                         image_file,
            # voter info
            "registered_num":                     vi.get("Registered_num", ""),
            "accredited_num":                     vi.get("Accredited_num", ""),
            "accredit_mark":                      vi.get("Accredit_mark", ""),
            "APC":                                vi.get("APC", ""),
            "PDP":                                vi.get("PDP", ""),
            "LP":                                 vi.get("LP", ""),
            "NNPP":                               vi.get("NNPP", ""),
            "total_use":                          vi.get("total_use", ""),
            # stamp / signature
            "presiding_officer_name_present":     ss.get("presiding_officer_name_present", ""),
            "presiding_officer_signature_present":ss.get("presiding_officer_signature_present", ""),
            "polling_agent_signature_present":    ss.get("polling_agent_signature_present", ""),
            "black_stamp":                        ss.get("black_stamp", ""),
        })

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    annotations_path = out_dir / "annotations.csv"
    with open(annotations_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=annotation_fields)
        writer.writeheader()
        writer.writerows(annotation_rows)

    if failed_rows:
        failed_path = out_dir / "failed.csv"
        with open(failed_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["polling_unit_code", "reason"])
            writer.writeheader()
            writer.writerows(failed_rows)
        print(f"\n  {len(failed_rows)} failures logged to {failed_path}")

    print(f"\nDone. {len(annotation_rows)} images saved to {images_dir}/")
    print(f"Annotations written to {annotations_path}")


if __name__ == "__main__":
    main()
