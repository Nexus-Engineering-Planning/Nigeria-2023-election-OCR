"""
demo/server.py — E Don Kast ADC 2026 Hackathon Demo Server

Self-contained Flask app. No database — results live in memory for the
duration of the demo session.

Routes:
  GET  /              → mobile upload page (audience scans QR, photographs EC8A form)
  GET  /dashboard     → live dashboard (projected on screen)
  POST /submit        → receives image → Claude Vision → stores result → returns JSON
  GET  /results       → JSON of all results so far (no base64 blobs)
  GET  /thumb/<id>    → 80px JPEG thumbnail for a submitted form
  GET  /preview/<id>  → 600px JPEG preview for a submitted form (lightbox)
  POST /reset         → clears all results (run before demo starts)

Setup:
  pip install flask anthropic pillow

Run:
  cd Nigeria-2023-election-OCR/demo
  python server.py
"""

import anthropic
import base64
import io
import json
import os
import threading
import uuid
from datetime import datetime

from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ── Upload size limit (15 MB) ─────────────────────────────────────────────────
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Image too large — please use a photo under 15 MB"}), 413


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

# ── In-memory stores ──────────────────────────────────────────────────────────
_lock    = threading.Lock()
_results = []          # list of result dicts — NO base64 blobs here
_media   = {}          # id → {"thumb": bytes, "preview": bytes}

# ── Anthropic client (singleton — created lazily on first request) ────────────
_anthropic_client: anthropic.Anthropic | None = None
_client_lock = threading.Lock()


def get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        with _client_lock:
            if _anthropic_client is None:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
                _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


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


# ── Image helpers ─────────────────────────────────────────────────────────────

def _normalize_image(image_bytes: bytes) -> Image.Image:
    """Open image bytes, flatten transparency, return RGB PIL Image."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("P", "RGBA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        src = img.convert("RGBA") if img.mode == "P" else img
        bg.paste(src, mask=src.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def make_thumbnails(image_bytes: bytes) -> tuple[bytes, bytes]:
    """
    Decode image once, return (thumb_jpeg_bytes, preview_jpeg_bytes).
    thumb  = 80px wide   (feed row)
    preview = 600px wide (lightbox)
    """
    img = _normalize_image(image_bytes)

    def _resize_encode(src: Image.Image, max_w: int) -> bytes:
        if src.width > max_w:
            ratio = max_w / src.width
            src = src.resize((max_w, int(src.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        src.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    return _resize_encode(img, 80), _resize_encode(img, 600)


def image_to_base64_jpeg(image_bytes: bytes) -> str:
    """Convert raw image bytes → base64 JPEG string (full resolution for Claude)."""
    img = _normalize_image(image_bytes)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def extract_with_claude(image_bytes: bytes) -> dict:
    client = get_anthropic_client()
    img_b64 = image_to_base64_jpeg(image_bytes)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
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

    # Generate thumbnails (single decode) before the slower Claude call
    try:
        thumb_bytes, preview_bytes = make_thumbnails(image_bytes)
    except Exception:
        thumb_bytes = preview_bytes = None

    try:
        extraction = extract_with_claude(image_bytes)
    except json.JSONDecodeError:
        return jsonify({"error": "Could not read the form — try a clearer photo."}), 500
    except Exception as e:
        # Log internally but never expose raw exception text to the client
        import logging
        logging.exception("Extraction error")
        msg = str(e).lower()
        if "api_key" in msg or "authentication" in msg or "401" in msg:
            return jsonify({"error": "Service configuration issue — contact the demo organiser."}), 500
        return jsonify({"error": "Verification failed — please retake and try again."}), 500

    parties = extraction.get("parties", {})
    flags   = extraction.get("flags",   {})
    meta    = extraction.get("meta",    {})
    summary = extraction.get("summary", {})

    record_id = str(uuid.uuid4())

    record = {
        "id":           record_id,
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
        "has_media":    thumb_bytes is not None,
    }

    with _lock:
        _results.append(record)
        if thumb_bytes and preview_bytes:
            _media[record_id] = {
                "thumb":   thumb_bytes,
                "preview": preview_bytes,
            }

    return jsonify(extraction)


@app.route("/thumb/<record_id>")
def thumb(record_id: str):
    with _lock:
        media = _media.get(record_id)
    if not media:
        return Response(status=404)
    return Response(
        media["thumb"],
        mimetype="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.route("/preview/<record_id>")
def preview(record_id: str):
    with _lock:
        media = _media.get(record_id)
    if not media:
        return Response(status=404)
    return Response(
        media["preview"],
        mimetype="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


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
        _media.clear()
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
