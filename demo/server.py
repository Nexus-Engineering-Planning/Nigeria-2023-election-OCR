"""
demo/server.py — E Don Kast ADC 2026 Hackathon Demo Server

Self-contained Flask app. No database — results live in memory for the
duration of the demo session.

Routes:
  GET  /              → mobile upload page (audience scans QR, photographs EC8A form)
  GET  /dashboard     → live dashboard (projected on screen)
  POST /submit        → receives image → AI → stores result → returns JSON
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
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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

# ── Per-IP rate limiting ──────────────────────────────────────────────────────
_rate_limit: dict[str, float] = {}
_rate_limit_lock = threading.Lock()
SUBMIT_COOLDOWN = 3.0  # seconds between submissions per IP


def _is_rate_limited(ip: str) -> bool:
    """Return True if this IP submitted within the last SUBMIT_COOLDOWN seconds."""
    now = time.monotonic()
    with _rate_limit_lock:
        last = _rate_limit.get(ip, 0.0)
        if now - last < SUBMIT_COOLDOWN:
            return True
        _rate_limit[ip] = now
        return False

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
    "valid_votes_summary": null,
    "valid_votes_table": null,
    "rejected_ballots": null
  },
  "parties": {
    "APC":  { "figures": null, "words": "", "mismatch": false, "reconciled": null },
    "LP":   { "figures": null, "words": "", "mismatch": false, "reconciled": null },
    "PDP":  { "figures": null, "words": "", "mismatch": false, "reconciled": null },
    "NNPP": { "figures": null, "words": "", "mismatch": false, "reconciled": null },
    "other_parties_total": null
  },
  "flags": {
    "quality": "good",
    "stamp_obscures": false,
    "rotation": false,
    "unreadable_fields": [],
    "sum_valid_votes": null,
    "sum_party_votes": null,
    "sum_delta": null
  }
}

Rules:
- Use null for any field you cannot read with confidence.
- For party votes: prefer the figures column. If obscured, use the words column.
- If a party's votes box appears empty or blank (no digits written), record it as 0, not null.
  Note: receiving 0 votes is common — a blank figures cell means 0, not "unknown".
  Use null only when the cell clearly has a number written in it that you cannot decipher.
- Zero-figure override: If a party's figures column is blank/reads as 0 BUT the words column
  clearly states a positive number (e.g. 'two hundred and twenty nine'), the blank cell was
  misidentified as zero — the party actually received votes. In this case, skip Step 2 for this
  party and immediately set reconciled = words-value.
  Example: LP.figures=0, LP.words='two hundred and twenty nine'=229 → LP.reconciled=229.
  Exception: if words = 'NOUGHT', 'NIL', 'ZERO', or any zero-word, both columns agree on 0.
- Words column bleed: the IN WORDS cell is narrow and adjacent to the polling agent signature
  column. Text from outside the cell (names, signatures, stray words) may bleed in. When reading
  the words column, extract only the leading number-words (e.g. "one hundred and twenty three").
  Stop at the first word that is clearly not part of a number (e.g. a person's name, "VOTE",
  "VOTES", "ONLY", a signature). Ignore anything after that point.
  If after stripping bleed text the words column is empty, treat it as unreadable (null words).
- Number-word misspellings: Nigerian EC8A forms are handwritten and often contain non-standard
  spellings. Parse these phonetic variants as their intended numbers:
  "NINTY" or "NINETY" = 90, "FOURTY" or "FORTY" = 40, "EIGHTY" = 80, "SEVENTY" = 70,
  "NOUGHT" or "NAUGHT" = 0, "HUNDERED" or "HUNDERD" = hundred.
  Apply the same principle to any word that clearly sounds like a number despite irregular spelling.

Step 1 — Per-field cross-check (figures vs words) + valid_votes dual check:
- For each party: compare figures vs words column as before (mismatch=true/false).
- For valid_votes: the total appears in TWO places on this form — read both:
    • valid_votes_summary = the figure in numbered row 7 at the top of the form
    • valid_votes_table   = the figure in the "TOTAL VALID VOTES" row at the bottom of the party table
  Reconcile them:
    - If they agree → use that value as the target total.
    - If they disagree → use the one closer to the sum of party votes you can already read.
    - If both are unreadable → skip Step 2 entirely.

Step 2 — Sum-based reconciliation across all parties:
NOTE: Step 2 OVERRIDES the "prefer figures" rule above. Use whichever of figures or words
minimizes |candidate_sum − valid_votes|, even if that means choosing words over figures.
- Use the reconciled valid_votes from Step 1 as the target total.
- For each party with mismatch=true you have two candidate values: figures and the words-parsed number.
  Parties with mismatch=false have one fixed value.
- IMPORTANT: other_parties_total must be read directly from the form by summing the individually
  listed unlisted-party rows. Do NOT back-calculate it as (valid_votes − LP − APC − PDP − NNPP).
  Back-calculating cascades misreads: if LP is wrong, other_parties_total absorbs the error and
  falsely validates the wrong LP value. Read each minor party's figures cell independently.
- Enumerate every combination of figures/words choices across the mismatched parties.
  For each combination compute: candidate_sum = LP + APC + PDP + NNPP + other_parties_total (null = 0).
- Choose the combination whose candidate_sum is closest to the reconciled valid_votes.
  Tiebreak: prefer words over figures. When two combinations give equal candidate_sum distance,
  always choose the one that uses the words-parsed value, not the figures value.
  Examples:
  • Tie: LP_figures=149, LP_words=179, both give candidate_sum=292 → set LP.reconciled=179.
  • Clear sum winner: LP_figures=10, LP_words='NINTY' (=90), valid_votes=95, other parties=5.
    LP=10 gives candidate_sum=15 (delta=80). LP=90 gives candidate_sum=95 (delta=0).
    LP=90 wins — set LP.reconciled=90. The sum proves the figures reading is wrong.
  • Narrow winner despite large absolute delta: LP=343 (fixed), APC_figures=28, APC_words=23,
    valid_votes=252, others=15. APC=28: sum=386, delta=134. APC=23: sum=381, delta=129.
    APC=23 wins (129 < 134) — choose words even though both are far from valid_votes.
- Set each party's reconciled = the value chosen by the winning combination.
  For parties with mismatch=false, reconciled = figures.
- Set sum_valid_votes = the reconciled valid_votes used as target.
  Set sum_party_votes = sum of all reconciled values (treat null as 0).
  Set sum_delta = abs(sum_party_votes - sum_valid_votes), or null if either total is null.

Step 3 — Overflow correction (run if sum_party_votes > sum_valid_votes):
A) Hard overflow (sum_delta > 15): A party's reconciled value is "impossible" if it exceeds
   the remaining budget:
     budget_for_P = sum_valid_votes - sum_of_all_other_parties_reconciled
   If any party P has reconciled > budget_for_P AND budget_for_P >= 0:
   - P's value was likely misread with an extra digit (e.g. '4' recorded as '44').
   - Set P.reconciled = null and add P to unreadable_fields.
   - Update sum_party_votes and sum_delta accordingly.
   Apply this check to each party independently (largest offender first if multiple).

B) Stray-mark removal — MANDATORY: After Step 3A (or if sum_delta ≤ 15), run this check.
   Loop through every party P in this order: APC, LP, PDP, NNPP, other_parties_total.
   For each P, test ALL THREE conditions:
     1. P.words is null or empty string  (no written-out confirmation of this number)
     2. P.reconciled > 0
     3. sum_party_votes - P.reconciled == sum_valid_votes  (removing P hits the target exactly)
   If all three hold for any P: set P.reconciled = 0, update sum_party_votes, update sum_delta, STOP.
   Worked example (must match this logic exactly):
     APC.reconciled=3, APC.words='', LP=71, others=1. sum_party=75, sum_valid=72, sum_delta=3.
     Check APC: words='' ✓, reconciled=3>0 ✓, 75-3=72=sum_valid ✓ → ALL THREE hold.
     → Set APC.reconciled=0. sum_party=72, sum_delta=0. STOP.

- quality = "good" if most fields readable; "poor" if stamp/blur affects some; "unreadable" if mostly unreadable.
- unreadable_fields: list plain-English descriptions of what you could NOT read — e.g. "polling unit code", "ward name", "stamp", "APC votes", "unused ballots". Never use JSON field names or underscores.
- Return ONLY the JSON object."""


# ── Image helpers ─────────────────────────────────────────────────────────────

def _normalize_image(image_bytes: bytes) -> Image.Image:
    """Open image bytes, flatten transparency, return RGB PIL Image."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
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

    result = json.loads(raw.strip())
    for key in ("meta", "summary", "parties", "flags"):
        if key not in result:
            logger.warning("Claude response missing expected key: %s", key)
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def upload_page():
    return send_from_directory(".", "upload.html")


@app.route("/dashboard")
def dashboard_page():
    return send_from_directory(".", "dashboard.html")


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/submit", methods=["POST"])
def submit():
    ip = request.remote_addr or "unknown"
    if _is_rate_limited(ip):
        return jsonify({"error": "Please wait a moment before submitting again."}), 429

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
        logger.exception("Extraction error")
        msg = str(e).lower()
        if "api_key" in msg or "authentication" in msg or "401" in msg:
            return jsonify({"error": "Service configuration issue — contact the demo organiser."}), 500
        return jsonify({"error": "Verification failed — please retake and try again."}), 500

    parties = extraction.get("parties", {})
    flags   = extraction.get("flags",   {})
    meta    = extraction.get("meta",    {})
    summary = extraction.get("summary", {})

    record_id = str(uuid.uuid4())

    def _best(party_data: dict):
        """Return reconciled value if present, else fall back to figures."""
        v = party_data.get("reconciled")
        return v if v is not None else party_data.get("figures")

    lp_val   = _best(parties.get("LP",   {}))
    apc_val  = _best(parties.get("APC",  {}))
    pdp_val  = _best(parties.get("PDP",  {}))
    nnpp_val = _best(parties.get("NNPP", {}))

    # Duplicate detection — same LP + APC + PDP votes = almost certainly the same form
    duplicate = False
    with _lock:
        for existing in _results:
            if (
                lp_val  is not None and lp_val  == existing.get("lp") and
                apc_val is not None and apc_val == existing.get("apc") and
                pdp_val == existing.get("pdp")
            ):
                duplicate = True
                break

    record = {
        "id":           record_id,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "polling_unit": meta.get("polling_unit", ""),
        "pu_code":      meta.get("pu_code", ""),
        "lp":           lp_val,
        "apc":          apc_val,
        "pdp":          pdp_val,
        "nnpp":         nnpp_val,
        "lp_mismatch":  parties.get("LP",  {}).get("mismatch", False),
        "apc_mismatch": parties.get("APC", {}).get("mismatch", False),
        "sum_delta":    flags.get("sum_delta"),
        "accredited":   summary.get("accredited_voters"),
        "quality":      flags.get("quality", "unknown"),
        "flags":        ", ".join(flags.get("unreadable_fields", [])),
        "has_media":    thumb_bytes is not None,
        "duplicate":    duplicate,
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

    lp_total  = sum(r["lp"]  or 0 for r in data if not r.get("duplicate"))
    apc_total = sum(r["apc"] or 0 for r in data if not r.get("duplicate"))

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
