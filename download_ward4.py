"""
download_ward4.py

Downloads EC8A images for Ward 4 (RUMUODOMAYA, Obio-Akpor LGA, Rivers State)
from DocumentCloud for the ADC 2026 Hackathon demo.

Ward 4 = RA code 04, PU codes 32/15/04/xxx
LP dominated on EC8A forms; INEC declared APC winner — the core demo discrepancy.

Output: demo_images/  (flat folder of GIF files)

Usage:
    py download_ward4.py
    py download_ward4.py --limit 15   # first N clean polling units only
"""

import argparse
import csv
import time
import urllib.request
from pathlib import Path

POLLING_UNITS_CSV = "AllPollingUnitsInfo.csv"
VOTER_INFO_CSV    = "voter_info.csv"
IMAGE_URL_TEMPLATE = (
    "https://assets.documentcloud.org/documents/{doc_id}/pages/{slug}-p1-large.gif"
)

# INEC officially declared LGA-level result (LGALevelResult.csv)
INEC_DECLARED = {"APC": 80239, "LP": 3829, "PDP": 368, "NNPP": 161}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit",   type=int, default=None,
                        help="Max polling units to download (default: all clean)")
    parser.add_argument("--output",  type=str, default="demo_images",
                        help="Output folder (default: demo_images)")
    parser.add_argument("--delay",   type=float, default=0.3,
                        help="Seconds between requests (default: 0.3)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    # Load all ward 4 rows
    ward4 = []
    with open(POLLING_UNITS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            puc = row["polling_unit_code"].strip()
            if not puc.startswith("32/15/04/"):
                continue
            if row["status"] != "exist and not blur":
                continue
            if not row["URL"].strip():
                continue
            ward4.append(row)

    # Load voter info for LP/APC ground truth
    voter = {}
    with open(VOTER_INFO_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            voter[r["polling_unit_code"]] = r

    # Sort by PU number so we get PU001, PU002, etc.
    ward4.sort(key=lambda r: r["polling_unit_code"])

    if args.limit:
        ward4 = ward4[: args.limit]

    print(f"Ward 4 — RUMUODOMAYA (3A), Obio-Akpor, Rivers State")
    print(f"Polling units to download: {len(ward4)}")
    print(f"Output folder: {out_dir}/")
    print()

    # Download
    manifest = []
    ok_count = fail_count = 0

    for i, row in enumerate(ward4, 1):
        puc   = row["polling_unit_code"].strip()
        raw_id = row.get("id", "").strip()
        url    = row["URL"].strip()

        try:
            doc_id = str(int(float(raw_id)))
        except (ValueError, TypeError):
            print(f"[{i:>3}] {puc} — bad id, skipping")
            fail_count += 1
            continue

        last_seg = url.rstrip("/").split("/")[-1]
        slug = last_seg[len(doc_id) + 1:]
        if not slug:
            print(f"[{i:>3}] {puc} — bad slug, skipping")
            fail_count += 1
            continue

        img_url  = IMAGE_URL_TEMPLATE.format(doc_id=doc_id, slug=slug)
        filename = f"{puc.replace('/', '_')}_{doc_id}.gif"
        dest     = out_dir / filename

        vi  = voter.get(puc, {})
        lp  = vi.get("LP",  "?")
        apc = vi.get("APC", "?")

        if dest.exists():
            print(f"[{i:>3}] {puc}  LP={lp:>5}  APC={apc:>4}  — cached")
            ok_count += 1
        else:
            try:
                time.sleep(args.delay)
                req = urllib.request.Request(
                    img_url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                if len(data) < 500:
                    raise ValueError("Response too small")
                with open(dest, "wb") as f:
                    f.write(data)
                print(f"[{i:>3}] {puc}  LP={lp:>5}  APC={apc:>4}  — ok")
                ok_count += 1
            except Exception as e:
                print(f"[{i:>3}] {puc}  — FAILED: {e}")
                fail_count += 1
                continue

        manifest.append({
            "polling_unit_code": puc,
            "unit_name":         row.get("unit_name", ""),
            "doc_id":            doc_id,
            "filename":          filename,
            "lp_ground_truth":   lp,
            "apc_ground_truth":  apc,
        })

    # Write manifest CSV
    manifest_path = out_dir / "manifest.csv"
    import csv as csv_mod
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(
            f, fieldnames=["polling_unit_code", "unit_name", "doc_id",
                           "filename", "lp_ground_truth", "apc_ground_truth"]
        )
        writer.writeheader()
        writer.writerows(manifest)

    # Summary
    lp_gt  = sum(float(m["lp_ground_truth"])  for m in manifest if m["lp_ground_truth"]  not in ("", "?"))
    apc_gt = sum(float(m["apc_ground_truth"]) for m in manifest if m["apc_ground_truth"] not in ("", "?"))

    print()
    print(f"Downloaded: {ok_count}  |  Failed: {fail_count}")
    print(f"Manifest:   {manifest_path}")
    print()
    print("=== WARD 4 GROUND TRUTH (EC8A forms) ===")
    print(f"  LP : {int(lp_gt):,}")
    print(f"  APC: {int(apc_gt):,}")
    print()
    print("=== INEC DECLARED (full LGA) ===")
    print(f"  LP : {INEC_DECLARED['LP']:,}")
    print(f"  APC: {INEC_DECLARED['APC']:,}")


if __name__ == "__main__":
    main()
