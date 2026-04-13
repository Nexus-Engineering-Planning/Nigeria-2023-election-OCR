"""
build_training_data.py

Downloads INEC election result sheet images from DocumentCloud and pairs
them with ground-truth labels from the existing CSVs.

Features
--------
- --all          Download every polling unit that has a URL (~169 k images, all statuses)
- --sample N     Download a stratified sample of N images (default: 500)
- Resumable      Already-downloaded images are skipped automatically
- Concurrent     Uses a thread pool for parallel downloads (--workers, default 8)
- Append mode    Re-runs extend annotations.csv rather than overwriting it
- Retry          Each image is attempted up to --retries times (default 3)

Output structure
----------------
    training_data/
        images/          .gif page images (one per polling unit)
        annotations.csv  image filename + all available ground-truth labels
        failed.csv       records that failed after all retries

Usage
-----
    # Download everything with a URL (~169 k images, fully resumable)
    py build_training_data.py --all

    # Download a 2 000-image stratified sample
    py build_training_data.py --sample 2000

    # Use 16 threads, faster delay
    py build_training_data.py --all --workers 16 --delay 0.05

Defaults: --sample 500, --workers 8, --delay 0.1, --retries 3
"""

import argparse
import csv
import os
import random
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLLING_UNITS_CSV = "AllPollingUnitsInfo.csv"
VOTER_INFO_CSV    = "voter_info.csv"
STAMP_SIG_CSV     = "stamp_sig_missing.csv"

IMAGE_URL_TEMPLATE = (
    "https://assets.documentcloud.org/documents/{doc_id}/pages/{slug}-p1-large.gif"
)

ANNOTATION_FIELDS = [
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


def stratified_sample(rows, group_col, n, seed):
    """Sample up to n rows spread evenly across unique values of group_col."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r[group_col]].append(r)

    per_group = max(1, n // len(groups))
    sampled = []
    for group_rows in groups.values():
        take = min(per_group, len(group_rows))
        sampled.extend(rng.sample(group_rows, take))

    # Top up to n if rounding left us short
    remaining = [r for r in rows if r not in set(map(id, sampled))]
    rng.shuffle(remaining)
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
    last_segment = url.rstrip("/").split("/")[-1]
    slug = last_segment[len(doc_id) + 1:]
    if not slug:
        return None, None
    return doc_id, slug


def download_image(doc_id, slug, dest_path, delay, retries):
    """Download page-1 GIF from DocumentCloud assets. Returns True on success."""
    url = IMAGE_URL_TEMPLATE.format(doc_id=doc_id, slug=slug)
    for attempt in range(1, retries + 1):
        try:
            time.sleep(delay)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < 500:
                raise ValueError("Response too small — likely an error page")
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            if attempt == retries:
                return False
            time.sleep(delay * 2 * attempt)
    return False


def load_existing_pucs(annotations_path):
    """Return set of polling_unit_codes already in annotations.csv."""
    if not annotations_path.exists():
        return set()
    existing = set()
    with open(annotations_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            puc = row.get("polling_unit_code", "").strip()
            if puc:
                existing.add(puc)
    return existing


def build_annotation_row(row, doc_id, slug, image_file, voter_info, stamp_sig):
    puc = row["polling_unit_code"].strip()
    vi  = voter_info.get(puc, {})
    ss  = stamp_sig.get(puc, {})
    return {
        "polling_unit_code":                  puc,
        "state_name":                         row["state_name"],
        "lga_name":                           row["lga_name"],
        "ward_name":                          row["ward_name"],
        "unit_name":                          row["unit_name"],
        "doc_id":                             doc_id,
        "slug":                               slug,
        "image_file":                         image_file,
        "registered_num":                     vi.get("Registered_num", ""),
        "accredited_num":                     vi.get("Accredited_num", ""),
        "accredit_mark":                      vi.get("Accredit_mark", ""),
        "APC":                                vi.get("APC", ""),
        "PDP":                                vi.get("PDP", ""),
        "LP":                                 vi.get("LP", ""),
        "NNPP":                               vi.get("NNPP", ""),
        "total_use":                          vi.get("total_use", ""),
        "presiding_officer_name_present":     ss.get("presiding_officer_name_present", ""),
        "presiding_officer_signature_present":ss.get("presiding_officer_signature_present", ""),
        "polling_agent_signature_present":    ss.get("polling_agent_signature_present", ""),
        "black_stamp":                        ss.get("black_stamp", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--all",    action="store_true", default=False,
                     help="Download ALL polling units with a URL (~169 k images, all statuses)")
    grp.add_argument("--sample", type=int, default=500,
                     help="Number of images to download via stratified sampling (default: 500)")
    parser.add_argument("--status-filter", action="store_true", default=False,
                        help="Restrict to 'exist and not blur' rows only (default: off — "
                             "downloads all rows that have a URL)")
    parser.add_argument("--output",  type=str,   default="training_data",
                        help="Output directory (default: training_data)")
    parser.add_argument("--delay",   type=float, default=0.1,
                        help="Seconds between requests per thread (default: 0.1)")
    parser.add_argument("--workers", type=int,   default=8,
                        help="Concurrent download threads (default: 8)")
    parser.add_argument("--retries", type=int,   default=3,
                        help="Retry attempts per image on failure (default: 3)")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="Random seed for stratified sampling (default: 42)")
    args = parser.parse_args()

    out_dir    = Path(args.output)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = out_dir / "annotations.csv"
    failed_path      = out_dir / "failed.csv"

    # -----------------------------------------------------------------------
    # Load CSVs
    # -----------------------------------------------------------------------
    print("Loading CSVs...")
    voter_info = load_csv_as_dict(VOTER_INFO_CSV, "polling_unit_code")
    stamp_sig  = load_csv_as_dict(STAMP_SIG_CSV,  "polling_unit_code")

    candidates = []
    skipped_no_url = 0
    skipped_status = 0
    with open(POLLING_UNITS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["URL"].strip():
                skipped_no_url += 1
                continue
            if args.status_filter and row["status"] != "exist and not blur":
                skipped_status += 1
                continue
            candidates.append(row)

    print(f"  {len(candidates):,} polling units with a downloadable URL")
    if skipped_no_url:
        print(f"  Skipped {skipped_no_url:,} rows with no URL (nothing to download)")
    if skipped_status:
        print(f"  Skipped {skipped_status:,} rows filtered by --status-filter")

    # -----------------------------------------------------------------------
    # Decide which units to process
    # -----------------------------------------------------------------------
    if args.all:
        target = candidates
        suffix = " (use --status-filter to restrict to 'exist and not blur' only)" if not args.status_filter else ""
        print(f"  Mode: --all ({len(target):,} units){suffix}")
    else:
        target = stratified_sample(candidates, "state_name", args.sample, args.seed)
        print(f"  Mode: --sample {args.sample} "
              f"({len(target):,} units across "
              f"{len(set(r['state_name'] for r in target))} states)")

    # -----------------------------------------------------------------------
    # Skip already-downloaded images (resumability)
    # -----------------------------------------------------------------------
    existing_pucs = load_existing_pucs(annotations_path)
    already_on_disk = sum(
        1 for r in target
        if (images_dir / f"{r['polling_unit_code'].strip().replace('/', '_')}_{str(int(float(r['id'].strip())))}.gif").exists()
    )
    todo = [r for r in target if r["polling_unit_code"].strip() not in existing_pucs]
    print(f"  Already annotated: {len(existing_pucs):,} | "
          f"On disk: {already_on_disk:,} | "
          f"To download: {len(todo):,}")

    if not todo:
        print("\nNothing to do — all images already downloaded and annotated.")
        return

    # Estimated time
    est_secs = len(todo) * args.delay / args.workers
    h, m = divmod(int(est_secs), 3600)
    m //= 60
    print(f"  Estimated time @ {args.workers} workers × {args.delay}s delay: "
          f"~{h}h {m:02d}m\n")

    # -----------------------------------------------------------------------
    # Concurrent downloads
    # -----------------------------------------------------------------------
    annotation_rows = []
    failed_rows     = []
    lock            = threading.Lock()
    counter         = {"done": 0, "ok": 0, "fail": 0}
    total           = len(todo)

    def process(row):
        puc = row["polling_unit_code"].strip()
        doc_id, slug = doc_id_and_slug_from_row(row)

        if not doc_id:
            return None, {"polling_unit_code": puc, "reason": "missing id/slug"}

        image_file = f"{puc.replace('/', '_')}_{doc_id}.gif"
        dest_path  = images_dir / image_file

        if dest_path.exists():
            ok = True        # already on disk, just build annotation
        else:
            ok = download_image(doc_id, slug, dest_path, args.delay, args.retries)

        if ok:
            ann_row = build_annotation_row(row, doc_id, slug, image_file,
                                           voter_info, stamp_sig)
            return ann_row, None
        else:
            return None, {"polling_unit_code": puc, "reason": "download error"}

    # Open annotations.csv in append mode so partial runs accumulate safely
    ann_file_exists = annotations_path.exists()
    ann_fh = open(annotations_path, "a", newline="", encoding="utf-8")
    ann_writer = csv.DictWriter(ann_fh, fieldnames=ANNOTATION_FIELDS)
    if not ann_file_exists:
        ann_writer.writeheader()

    # Open failed.csv similarly
    fail_file_exists = failed_path.exists()
    fail_fh = open(failed_path, "a", newline="", encoding="utf-8")
    fail_writer = csv.DictWriter(fail_fh, fieldnames=["polling_unit_code", "reason"])
    if not fail_file_exists:
        fail_writer.writeheader()

    print_interval = max(1, min(100, total // 50))   # ~50 progress lines total

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, row): row for row in todo}
            for future in as_completed(futures):
                ann_row, fail_row = future.result()

                with lock:
                    counter["done"] += 1
                    if ann_row:
                        ann_writer.writerow(ann_row)
                        ann_fh.flush()
                        counter["ok"] += 1
                    if fail_row:
                        fail_writer.writerow(fail_row)
                        fail_fh.flush()
                        counter["fail"] += 1

                    done = counter["done"]
                    if done % print_interval == 0 or done == total:
                        pct = done / total * 100
                        print(f"  [{done:>6}/{total}] {pct:5.1f}%  "
                              f"ok={counter['ok']:,}  fail={counter['fail']:,}")
    finally:
        ann_fh.close()
        fail_fh.close()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_annotated = len(existing_pucs) + counter["ok"]
    print(f"\nDone.")
    print(f"  Downloaded this run : {counter['ok']:,}")
    print(f"  Failures this run   : {counter['fail']:,}")
    print(f"  Total in {annotations_path.name}: {total_annotated:,}")
    if counter["fail"]:
        print(f"  Failed list         : {failed_path}")
        print(f"  Tip: Re-run with same flags to retry failures "
              f"(already-downloaded images are skipped automatically).")


if __name__ == "__main__":
    main()
