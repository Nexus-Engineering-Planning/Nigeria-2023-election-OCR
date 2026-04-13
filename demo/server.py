"""
demo/server.py — E Don Kast ADC 2026 Hackathon Demo Server

Self-contained Flask app. No database — results live in memory for the
duration of the demo session.

Routes:
  GET  /              → mobile upload page (audience scans QR, photographs EC8A form)
  GET  /dashboard     → live dashboard (projected on screen)
  POST /submit        → receives image → Claude Vision → stores result → returns JSON
  GET  /results       → JSON of all results so far
  POST /reset         → clears all results (run before demo starts)

Setup:
  pip install flask anthropic pillow

Run:
  cd Nigeria-2023-election-OCR/demo
  python server.py

Expose for QR code (in a second terminal):
  ngrok http 5000
"""

import anthropic
import base64
import io
import json
import os
import threading
import uuid
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ── Ground truth ─────────────────────────────────────────────────────────────
# Ward 4 = RUMUODOMAYA (3A), Obio-Akpor LGA, Rivers State
# Source: voter_info.csv (EC8A polling unit aggregates) + LGALevelResult.csv

GROUND_TRUTH = {
    "ward":           "RUMUODOMAYA (Ward 4)",
    "lga":            "Obio-Akpor",
    "state":          "Rivers State",
    "ec8a_ward":      {"lp": 4432,  "apc": 388},    # this ward only (EC8A forms)
    "ec8a_lga":       {"lp": 68521, "apc": 16980},   # full LGA (EC8A forms)
    "inec_declared":  {"lp": 3829,  "apc": 80239},   # INEC officially declared (LGA)
}

# ── In-memory results store ───────────────────────────────────────────────────
_lock    = threading.Lock()
_results = []   # list of dicts, one per submitted form

# ── Claude extraction prompt ──────────────────────────────────────────────────
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
- Return ONLY the JSON object."""


def image_to_base64_jpeg(image_bytes: bytes) -> str:
    """Convert any image format to base64-encoded JPEG."""
    img = Image.open(io.BytesIO(image_bytes))
    img.seek(0) if hasattr(img, "seek") else None
    if img.mode in ("P", "RGBA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def extract_with_claude(image_bytes: bytes) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    client = anthropic.Anthropic(api_key=api_key)
    img_b64 = image_to_base64_jpeg(image_bytes)

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": "image/jpeg",
                        "data":       img_b64,
                    },
                },
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        }],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def upload_page():
    return send_from_directory(".", "upload.html")


@app.route("/dashboard")
def dashboard_page():
    return send_from_directory(".", "dashboard.html")


@app.route("/submit", methods=["POST"])
def submit():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    if len(image_bytes) < 500:
        return jsonify({"error": "Image too small or empty"}), 400

    try:
        extraction = extract_with_claude(image_bytes)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Could not parse Claude response: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    parties = extraction.get("parties", {})
    flags   = extraction.get("flags",   {})
    meta    = extraction.get("meta",    {})
    summary = extraction.get("summary", {})

    record = {
        "id":           str(uuid.uuid4()),
        "timestamp":    datetime.utcnow().isoformat(),
        "polling_unit": meta.get("polling_unit", ""),
        "pu_code":      meta.get("pu_code", ""),
        "lp":           parties.get("LP",   {}).get("figures"),
        "apc":          parties.get("APC",  {}).get("figures"),
        "pdp":          parties.get("PDP",  {}).get("figures"),
        "nnpp":         parties.get("NNPP", {}).get("figures"),
        "accredited":   summary.get("accredited_voters"),
        "quality":      flags.get("quality", "unknown"),
        "flags":        ", ".join(flags.get("unreadable_fields", [])),
    }

    with _lock:
        _results.append(record)

    return jsonify(extraction)


@app.route("/results")
def results():
    with _lock:
        data = list(_results)

    lp_total  = sum(r["lp"]  or 0 for r in data)
    apc_total = sum(r["apc"] or 0 for r in data)

    return jsonify({
        "results":      data,
        "count":        len(data),
        "totals":       {"lp": lp_total, "apc": apc_total},
        "ground_truth": GROUND_TRUTH,
    })


@app.route("/reset", methods=["POST"])
def reset():
    with _lock:
        _results.clear()
    return jsonify({"ok": True})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nWARNING: ANTHROPIC_API_KEY not set — /submit will return errors.")
        print("Set it before submitting forms:")
        print("  PowerShell: $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
        print("  Railway/Render: add as environment variable in dashboard\n")

    print("\n" + "="*55)
    print("  E Don Kast — ADC 2026 Hackathon Demo")
    print("="*55)
    print(f"  Ward:      {GROUND_TRUTH['ward']}")
    print(f"  LGA:       {GROUND_TRUTH['lga']}, {GROUND_TRUTH['state']}")
    print()
    print(f"  Upload page:  http://localhost:{port}/")
    print(f"  Dashboard:    http://localhost:{port}/dashboard")
    print("="*55 + "\n")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
