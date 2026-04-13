#!/usr/bin/env python3
"""
EC8A Form Extraction Prototype
--------------------------------
Reads INEC EC8A result sheet images using Claude Vision
and extracts structured data into a CSV.

Usage:
    python ec8a_extract.py --input /path/to/images --output results.csv
    
    Set your API key first:
    export ANTHROPIC_API_KEY=your_key_here
"""

import anthropic
import base64
import csv
import json
import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import io

# ── Supported image extensions ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# ── The extraction prompt ────────────────────────────────────────────────────
EXTRACTION_PROMPT = """You are extracting data from a Nigerian INEC EC8A election result sheet (Form EC8A - Statement of Result of Poll from Polling Unit, 2023 Presidential Election).

Extract ALL of the following fields and return ONLY valid JSON, nothing else.

{
  "meta": {
    "state": "",
    "lga": "",
    "registration_area": "",
    "polling_unit": "",
    "state_code": "",
    "lga_code": "",
    "ra_code": "",
    "pu_code": "",
    "serial_number": "",
    "date": ""
  },
  "summary": {
    "voters_on_register": null,
    "accredited_voters": null,
    "ballot_papers_issued": null,
    "unused_ballot_papers": null,
    "spoiled_ballot_papers": null,
    "rejected_ballots": null,
    "total_valid_votes": null,
    "total_used_ballot_papers": null
  },
  "party_results": {
    "A": {"figures": null, "words": ""},
    "AA": {"figures": null, "words": ""},
    "AAC": {"figures": null, "words": ""},
    "ADC": {"figures": null, "words": ""},
    "ADP": {"figures": null, "words": ""},
    "APC": {"figures": null, "words": ""},
    "APGA": {"figures": null, "words": ""},
    "APM": {"figures": null, "words": ""},
    "APP": {"figures": null, "words": ""},
    "BP": {"figures": null, "words": ""},
    "LP": {"figures": null, "words": ""},
    "NNPP": {"figures": null, "words": ""},
    "NRM": {"figures": null, "words": ""},
    "PDP": {"figures": null, "words": ""},
    "PRP": {"figures": null, "words": ""},
    "SDP": {"figures": null, "words": ""},
    "YPP": {"figures": null, "words": ""},
    "ZLP": {"figures": null, "words": ""}
  },
  "flags": {
    "image_quality": "good|poor|unreadable",
    "stamp_obscures_data": false,
    "rotation_issue": false,
    "fields_unreadable": []
  }
}

Rules:
- Use null for any numeric field you cannot read, not 0
- Use empty string "" for any text field you cannot read
- For "words" fields, record exactly what is written (e.g. "NIL", "EIGHT", "ONE HUNDRED")
- For "figures" fields, return an integer or null
- In flags.fields_unreadable, list any field names you could not read
- image_quality: "good" if most fields readable, "poor" if degraded but partially readable, "unreadable" if you cannot extract meaningful data
- Return ONLY the JSON object. No explanation, no markdown, no preamble."""


def load_image_as_base64(image_path: Path) -> tuple[str, str]:
    """Load an image and convert to base64. Returns (base64_data, media_type)."""
    suffix = image_path.suffix.lower()
    
    # GIFs need to be converted to PNG for the API
    if suffix == ".gif":
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            data = base64.standard_b64encode(buffer.read()).decode("utf-8")
            return data, "image/png"
    
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def extract_form(client: anthropic.Anthropic, image_path: Path) -> dict:
    """Send one image to Claude and return parsed extraction result."""
    print(f"  Processing: {image_path.name}")
    
    try:
        img_data, media_type = load_image_as_base64(image_path)
    except Exception as e:
        return {"error": f"Failed to load image: {e}", "filename": image_path.name}

    try:
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": EXTRACTION_PROMPT
                        }
                    ],
                }
            ],
        )
        
        raw = message.content[0].text.strip()
        
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        result = json.loads(raw)
        result["filename"] = image_path.name
        return result
    
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parse error: {e}",
            "raw_response": raw[:500] if 'raw' in locals() else "",
            "filename": image_path.name
        }
    except Exception as e:
        return {"error": str(e), "filename": image_path.name}


def flatten_result(result: dict) -> dict:
    """Flatten nested JSON into a flat dict for CSV output."""
    if "error" in result:
        return {
            "filename": result.get("filename", ""),
            "error": result["error"],
        }
    
    flat = {"filename": result.get("filename", "")}
    
    # Meta fields
    for k, v in result.get("meta", {}).items():
        flat[f"meta_{k}"] = v
    
    # Summary fields
    for k, v in result.get("summary", {}).items():
        flat[f"summary_{k}"] = v
    
    # Party results
    for party, data in result.get("party_results", {}).items():
        flat[f"party_{party}_figures"] = data.get("figures")
        flat[f"party_{party}_words"] = data.get("words", "")
    
    # Flags
    flags = result.get("flags", {})
    flat["flag_quality"] = flags.get("image_quality", "")
    flat["flag_stamp_obscures"] = flags.get("stamp_obscures_data", False)
    flat["flag_rotation"] = flags.get("rotation_issue", False)
    flat["flag_unreadable_fields"] = "|".join(flags.get("fields_unreadable", []))
    
    return flat


def write_csv(rows: list[dict], output_path: Path):
    """Write flattened results to CSV."""
    if not rows:
        print("No results to write.")
        return
    
    # Collect all keys across all rows
    all_keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract EC8A form data using Claude Vision")
    parser.add_argument("--input", "-i", required=True, help="Folder containing EC8A images")
    parser.add_argument("--output", "-o", default="ec8a_results.csv", help="Output CSV path")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Max images to process (for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Run: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: Input folder not found: {input_dir}")
        sys.exit(1)

    # Find all supported images
    images = [
        p for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    
    if not images:
        print(f"No supported images found in {input_dir}")
        sys.exit(1)
    
    if args.limit:
        images = images[:args.limit]
    
    print(f"\nEC8A Extraction Prototype")
    print(f"{'─' * 40}")
    print(f"Input folder : {input_dir}")
    print(f"Images found : {len(images)}")
    print(f"Output file  : {args.output}")
    print(f"{'─' * 40}\n")

    client = anthropic.Anthropic(api_key=api_key)
    
    results = []
    raw_results = []
    
    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}]", end=" ")
        result = extract_form(client, image_path)
        raw_results.append(result)
        results.append(flatten_result(result))
        
        # Show a quick summary
        if "error" not in result:
            meta = result.get("meta", {})
            summary = result.get("summary", {})
            flags = result.get("flags", {})
            print(f"    State: {meta.get('state', '?')} | PU: {meta.get('polling_unit', '?')} | "
                  f"Valid votes: {summary.get('total_valid_votes', '?')} | "
                  f"Quality: {flags.get('image_quality', '?')}")
        else:
            print(f"    ERROR: {result['error']}")
    
    # Save CSV
    output_path = Path(args.output)
    write_csv(results, output_path)
    
    # Save raw JSON too (useful for debugging)
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Raw JSON saved to: {json_path}")
    
    # Summary stats
    errors = sum(1 for r in raw_results if "error" in r)
    good = sum(1 for r in raw_results if r.get("flags", {}).get("image_quality") == "good")
    poor = sum(1 for r in raw_results if r.get("flags", {}).get("image_quality") == "poor")
    
    print(f"\nSummary")
    print(f"{'─' * 40}")
    print(f"Processed  : {len(images)}")
    print(f"Errors     : {errors}")
    print(f"Good quality: {good}")
    print(f"Poor quality: {poor}")


if __name__ == "__main__":
    main()