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

# ── Optional dewarp import (scripts/dewarp.py) ───────────────────────────────
# Insert repo root so "from scripts.dewarp import ..." works regardless of cwd.
_REPO_ROOT = str(Path(__file__).parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from scripts.dewarp import (
        dewarp_to_bytes as _dewarp_to_bytes,
        load_models as _load_dewarp_models,
        DewarpError,
    )
    _DEWARP_AVAILABLE = True
except ImportError:
    _DEWARP_AVAILABLE = False
    _dewarp_to_bytes = None
    _load_dewarp_models = None

    class DewarpError(Exception):
        pass

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


def load_image_as_base64(
    image_path: Path,
    det_model=None,
    rect_model=None,
) -> tuple[str, str, bool]:
    """Load an image and return (base64_data, media_type, dewarp_failed).

    When det_model and rect_model are supplied, attempts to crop+dewarp the
    image first. On DewarpError (orientation indeterminate) or any other
    dewarping failure, falls back to the raw image and sets dewarp_failed=True.
    """
    dewarp_failed = False

    if det_model is not None and rect_model is not None and _DEWARP_AVAILABLE:
        try:
            jpeg_bytes = _dewarp_to_bytes(image_path, det_model, rect_model)
            data = base64.standard_b64encode(jpeg_bytes).decode("utf-8")
            return data, "image/jpeg", False
        except DewarpError as e:
            print(f"    [WARN] Dewarp orientation failed ({e}); using raw image")
            dewarp_failed = True
        except Exception as e:
            print(f"    [WARN] Dewarp error ({e}); using raw image")
            dewarp_failed = True

    # ── Raw loading fallback ──────────────────────────────────────────────────
    suffix = image_path.suffix.lower()

    # GIFs need to be converted to PNG for the API
    if suffix == ".gif":
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            data = base64.standard_b64encode(buffer.read()).decode("utf-8")
            return data, "image/png", dewarp_failed

    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type, dewarp_failed


def extract_form(
    client: anthropic.Anthropic,
    image_path: Path,
    source_channel: str,
    det_model=None,
    rect_model=None,
) -> dict:
    """Send one image to Claude and return parsed extraction result."""
    print(f"  Processing: {image_path.name}")

    try:
        img_data, media_type, dewarp_failed = load_image_as_base64(
            image_path, det_model, rect_model
        )
    except Exception as e:
        return {
            "error": f"Failed to load image: {e}",
            "filename": image_path.name,
            "image_source": source_channel,
            "dewarp_failed": False,
        }

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
        result["image_source"] = source_channel
        result["dewarp_failed"] = dewarp_failed
        return result

    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parse error: {e}",
            "raw_response": raw[:500] if 'raw' in locals() else "",
            "filename": image_path.name,
            "image_source": source_channel,
            "dewarp_failed": dewarp_failed,
        }
    except Exception as e:
        return {
            "error": str(e),
            "filename": image_path.name,
            "image_source": source_channel,
            "dewarp_failed": False,
        }


def flatten_result(result: dict) -> dict:
    """Flatten nested JSON into a flat dict for CSV output."""
    if "error" in result:
        return {
            "filename": result.get("filename", ""),
            "image_source": result.get("image_source", ""),
            "error": result["error"],
        }

    flat = {
        "filename": result.get("filename", ""),
        "image_source": result.get("image_source", ""),
    }

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
    flat["flag_dewarp_failed"] = result.get("dewarp_failed", False)

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
    parser.add_argument(
        "--source-channel", "-s",
        required=True,
        choices=["crowdsource", "scrape"],
        help=(
            "Image source channel used for routing and image_source labeling. "
            "'crowdsource' = citizen phone photos. "
            "'scrape' = IReV archive PDFs/scans. "
            "Dewarping is off by default for both; use --dewarp to enable for crowdsource."
        ),
    )
    parser.add_argument(
        "--dewarp",
        action="store_true",
        help="Enable dewarping for crowdsource images (opt-in; off by default).",
    )
    parser.add_argument("--input", "-i", required=True, help="Folder containing EC8A images")
    parser.add_argument("--output", "-o", default="ec8a_results.csv", help="Output CSV path")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Max images to process (for testing)")
    args = parser.parse_args()

    # Validate channel / dewarp combination
    if args.dewarp and args.source_channel == "scrape":
        parser.error(
            "--dewarp is not valid with --source-channel scrape "
            "(dewarping only applies to crowdsource images)"
        )

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

    # Load dewarp models once at startup (crowdsource only)
    det_model = None
    rect_model = None
    dewarp_active = args.dewarp and args.source_channel == "crowdsource"

    if dewarp_active:
        if not _DEWARP_AVAILABLE:
            print("WARNING: scripts/dewarp.py dependencies not available; running without dewarping.")
            dewarp_active = False
        else:
            print("Loading dewarp models...")
            try:
                det_model, rect_model = _load_dewarp_models()
                print("  Models loaded.")
            except Exception as e:
                print(f"WARNING: Failed to load dewarp models ({e}); running without dewarping.")
                det_model = rect_model = None
                dewarp_active = False

    print(f"\nEC8A Extraction Prototype")
    print(f"{'─' * 40}")
    print(f"Source channel : {args.source_channel}")
    print(f"Dewarp         : {'enabled' if dewarp_active else 'disabled'}")
    print(f"Input folder   : {input_dir}")
    print(f"Images found   : {len(images)}")
    print(f"Output file    : {args.output}")
    print(f"{'─' * 40}\n")

    client = anthropic.Anthropic(api_key=api_key)

    results = []
    raw_results = []

    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}]", end=" ")
        result = extract_form(
            client, image_path, args.source_channel, det_model, rect_model
        )
        raw_results.append(result)
        results.append(flatten_result(result))

        # Show a quick summary
        if "error" not in result:
            meta = result.get("meta", {})
            summary = result.get("summary", {})
            flags = result.get("flags", {})
            dewarp_note = " [dewarp-failed]" if result.get("dewarp_failed") else ""
            print(
                f"    State: {meta.get('state', '?')} | PU: {meta.get('polling_unit', '?')} | "
                f"Valid votes: {summary.get('total_valid_votes', '?')} | "
                f"Quality: {flags.get('image_quality', '?')}{dewarp_note}"
            )
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
    dewarp_failed_count = sum(1 for r in raw_results if r.get("dewarp_failed"))

    print(f"\nSummary")
    print(f"{'─' * 40}")
    print(f"Processed      : {len(images)}")
    print(f"Errors         : {errors}")
    print(f"Good quality   : {good}")
    print(f"Poor quality   : {poor}")
    if dewarp_active or dewarp_failed_count:
        print(f"Dewarp failed  : {dewarp_failed_count}")


if __name__ == "__main__":
    main()