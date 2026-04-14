"""
test_extraction_detail.py — Deep-dive on WRONG and PARTIAL cases.

Re-extracts only those images and prints the full Claude response
(figures + words for each party) to diagnose failure patterns.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import anthropic

IMAGES_DIR = Path(__file__).parent / "demo_images"
MANIFEST   = IMAGES_DIR / "manifest.csv"

# Indices (1-based, matching test_extraction.py output) to re-examine
TARGET_STATUSES = {"WRONG", "PARTIAL"}

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
    "valid_votes": null,
    "rejected_ballots": null
  },
  "parties": {
    "APC":  { "figures": null, "words": "" },
    "LP":   { "figures": null, "words": "" },
    "PDP":  { "figures": null, "words": "" },
    "NNPP": { "figures": null, "words": "" },
    "other_parties_total": null
  },
  "flags": {
    "quality": "good",
    "stamp_obscures": false,
    "rotation": false,
    "unreadable_fields": []
  }
}

Rules:
- Use null for any field you cannot read with confidence.
- For party votes: prefer the figures column. If obscured, use the words column.
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


def extract(client, img_b64):
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
    lp_match  = (ex_lp  == gt_lp)  if (ex_lp  is not None and gt_lp  is not None) else False
    apc_match = (ex_apc == gt_apc) if (ex_apc is not None and gt_apc is not None) else False
    if lp_match and apc_match:  return "EXACT"
    if lp_match or apc_match:   return "PARTIAL"
    if ex_lp is None and ex_apc is None: return "NULL"
    return "WRONG"


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # --- First pass: classify using previous run's known results ---
    # (re-extract to get fresh data including words field)
    known_wrong_partial = {
        "32/15/04/002", "32/15/04/009", "32/15/04/010",
        "32/15/04/024", "32/15/04/030",   # WRONG
        "32/15/04/005", "32/15/04/011", "32/15/04/012",
        "32/15/04/017", "32/15/04/018", "32/15/04/019",
        "32/15/04/023", "32/15/04/029", "32/15/04/036",
        "32/15/04/037", "32/15/04/038", "32/15/04/039",
        "32/15/04/045", "32/15/04/049", "32/15/04/050",
        "32/15/04/052", "32/15/04/053",  # PARTIAL
    }

    targets = [r for r in rows if r["polling_unit_code"] in known_wrong_partial]
    print(f"\nRe-extracting {len(targets)} WRONG/PARTIAL images with full detail...\n")

    wrong_summary  = []
    partial_summary = []

    for row in targets:
        pu_code  = row["polling_unit_code"]
        filename = row["filename"]
        gt_lp    = int(float(row["lp_ground_truth"]))  if row["lp_ground_truth"]  else None
        gt_apc   = int(float(row["apc_ground_truth"])) if row["apc_ground_truth"] else None

        img_path = IMAGES_DIR / filename
        try:
            img_b64 = load_image_b64(img_path)
            result  = extract(client, img_b64)
        except Exception as e:
            print(f"  {pu_code}: ERROR — {e}\n")
            time.sleep(0.5)
            continue

        parties  = result.get("parties", {})
        flags    = result.get("flags",   {})
        quality  = flags.get("quality", "?")
        unread   = flags.get("unreadable_fields", [])
        stamp    = flags.get("stamp_obscures", False)

        ex_lp    = parties.get("LP",  {}).get("figures")
        ex_apc   = parties.get("APC", {}).get("figures")
        lp_words = parties.get("LP",  {}).get("words", "")
        apc_words= parties.get("APC", {}).get("figures") # intentional — shows what was seen
        apc_words_str = parties.get("APC", {}).get("words", "")

        status = classify(ex_lp, ex_apc, gt_lp, gt_apc)

        lp_d  = f"{ex_lp  - gt_lp:+d}"  if (ex_lp  is not None and gt_lp  is not None) else "—"
        apc_d = f"{ex_apc - gt_apc:+d}" if (ex_apc is not None and gt_apc is not None) else "—"

        print(f"[{status}]  {pu_code}  ({filename})")
        print(f"  GT:        LP={gt_lp}   APC={gt_apc}")
        print(f"  Extracted: LP={ex_lp} (words: '{lp_words}')   APC={ex_apc} (words: '{apc_words_str}')")
        print(f"  Diff:      LP={lp_d}  APC={apc_d}")
        print(f"  Quality:   {quality}   stamp_obscures={stamp}")
        if unread:
            print(f"  Unreadable: {', '.join(unread)}")
        print()

        time.sleep(0.3)

    # --- Pattern summary ---
    print("=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)
    print("""
Common failure modes to look for:
  1. APC=null when GT is 0 or very small  -> Claude treats blank/zero as unreadable
  2. LP digit errors (e.g. 179->149)      -> stamp or blur over a digit
  3. Both values = 0                       -> image quality / rotation issue
  4. Transposition (142->104, 2->42)      -> misidentified row on the form
""")


if __name__ == "__main__":
    main()
