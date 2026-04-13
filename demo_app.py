"""
demo_app.py — E Don Kast ADC 2026 Hackathon Demo

A self-contained Flask app with three routes:
  /           → mobile upload page (audience scans QR, photographs their EC8A form)
  /submit     → POST endpoint: receives image, calls Claude Vision, returns JSON
  /dashboard  → live results dashboard (projected on screen)

Run:
    pip install flask anthropic pillow
    python demo_app.py

Then:
  Upload page:   http://localhost:5000/
  Dashboard:     http://localhost:5000/dashboard
  API endpoint:  http://localhost:5000/submit  (POST, multipart image)

For live demo — expose via ngrok:
    ngrok http 5000
    Use the ngrok URL for the QR code.
"""

import anthropic
import base64
import csv
import io
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from PIL import Image

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Ground truth — Ward 4 (RUMUODOMAYA), Obio-Akpor, Rivers State
# LP dominated on EC8A forms; APC declared winner by INEC.
# ---------------------------------------------------------------------------
WARD_NAME   = "RUMUODOMAYA (Ward 4)"
LGA_NAME    = "Obio-Akpor"
STATE_NAME  = "Rivers State"

# EC8A-level ground truth (voter_info.csv — polling unit totals across full LGA)
EC8A_LGA_TRUTH = {"LP": 68521, "APC": 16980}

# INEC officially declared (LGALevelResult.csv)
INEC_DECLARED  = {"LP": 3829,  "APC": 80239}

# In-memory results store (thread-safe)
_lock   = threading.Lock()
_results = []   # list of dicts, one per submission


# ---------------------------------------------------------------------------
# Claude Vision extraction (same logic as ec8a_extract.py)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are extracting data from a Nigerian INEC EC8A election result sheet.
This is a Form EC8A — Statement of Result of Poll from Polling Unit, 2023 Presidential Election.

Extract the following and return ONLY valid JSON — no prose, no markdown:

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
    "APC": {"figures": null, "words": ""},
    "LP":  {"figures": null, "words": ""},
    "PDP": {"figures": null, "words": ""},
    "NNPP":{"figures": null, "words": ""},
    "other_parties_total": null
  },
  "flags": {
    "quality": "good|poor|unreadable",
    "stamp_obscures": false,
    "rotation": false,
    "unreadable_fields": []
  }
}

Rules:
- Use null for any field you cannot read with reasonable confidence.
- For party votes: use the numeric figures column. If figures are obscured, use the words column.
- quality = "good" if most fields are readable; "poor" if stamp/rotation/blur affects some fields; "unreadable" if you cannot extract meaningful data.
- Return ONLY the JSON object — no explanation."""


def extract_from_image(image_bytes: bytes) -> dict:
    """Send image to Claude Vision and return structured extraction."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Detect format, convert to JPEG for API
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("P", "RGBA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.standard_b64encode(buf.getvalue()).decode()

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type":  "image",
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
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

UPLOAD_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>E Don Kast — Scan Your Form</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a1628;
    color: #fff;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }
  .logo { font-size: 28px; font-weight: 800; color: #00d4aa; margin-bottom: 8px; }
  .subtitle { color: #8899aa; font-size: 14px; margin-bottom: 40px; text-align: center; }
  .card {
    background: #162032;
    border-radius: 16px;
    padding: 32px 24px;
    width: 100%;
    max-width: 400px;
    text-align: center;
  }
  .instruction {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 8px;
    line-height: 1.4;
  }
  .sub {
    color: #8899aa;
    font-size: 13px;
    margin-bottom: 28px;
  }
  .upload-btn {
    display: block;
    width: 100%;
    padding: 18px;
    background: #00d4aa;
    color: #0a1628;
    border: none;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
    margin-bottom: 16px;
  }
  .upload-btn:active { opacity: 0.85; }
  input[type=file] { display: none; }
  #preview {
    display: none;
    width: 100%;
    border-radius: 8px;
    margin-bottom: 16px;
    max-height: 300px;
    object-fit: contain;
  }
  #submit-btn {
    display: none;
    width: 100%;
    padding: 18px;
    background: #1a7af4;
    color: #fff;
    border: none;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
  }
  #status {
    margin-top: 20px;
    font-size: 15px;
    min-height: 24px;
    color: #00d4aa;
  }
  .spinner { display: inline-block; animation: spin 1s linear infinite; }
  @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
  .ward-tag {
    display: inline-block;
    background: rgba(0,212,170,0.12);
    color: #00d4aa;
    font-size: 12px;
    border-radius: 6px;
    padding: 4px 10px;
    margin-bottom: 20px;
  }
</style>
</head>
<body>
  <div class="logo">E Don Kast</div>
  <div class="subtitle">Nigeria 2023 Election Verification</div>
  <div class="card">
    <div class="ward-tag">Ward 4 · Obio-Akpor · Rivers State</div>
    <div class="instruction">Photograph your EC8A form</div>
    <div class="sub">Hold steady, make sure all text is visible</div>

    <label for="file-input">
      <div class="upload-btn" id="camera-btn">📷 Take Photo / Upload</div>
    </label>
    <input type="file" id="file-input" accept="image/*" capture="environment">

    <img id="preview" alt="Form preview">
    <button class="upload-btn" id="submit-btn" style="background:#1a7af4;color:#fff;">
      ✓ Submit Form
    </button>

    <div id="status"></div>
  </div>

<script>
  const fileInput = document.getElementById('file-input');
  const preview   = document.getElementById('preview');
  const submitBtn = document.getElementById('submit-btn');
  const status    = document.getElementById('status');
  let selectedFile = null;

  fileInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    if (!selectedFile) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      preview.src = ev.target.result;
      preview.style.display = 'block';
      submitBtn.style.display = 'block';
    };
    reader.readAsDataURL(selectedFile);
  });

  submitBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    submitBtn.disabled = true;
    status.innerHTML = '<span class="spinner">⟳</span> Extracting results with Claude Vision...';

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const res = await fetch('/submit', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.error) {
        status.textContent = '⚠ ' + data.error;
        submitBtn.disabled = false;
      } else {
        const lp  = data.parties?.LP?.figures  ?? '—';
        const apc = data.parties?.APC?.figures ?? '—';
        status.innerHTML = `✅ Submitted!<br><strong>LP ${lp} · APC ${apc}</strong><br>
          <span style="color:#8899aa;font-size:13px">${data.meta?.polling_unit || ''}</span>`;
        submitBtn.style.display = 'none';
        preview.style.display   = 'none';
        document.getElementById('camera-btn').textContent = '📷 Submit Another Form';
      }
    } catch (err) {
      status.textContent = '⚠ Network error — please try again';
      submitBtn.disabled = false;
    }
  });
</script>
</body>
</html>"""


DASHBOARD_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>E Don Kast — Live Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a1628;
    color: #fff;
    padding: 32px;
    min-height: 100vh;
  }
  header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 32px;
  }
  .logo { font-size: 32px; font-weight: 800; color: #00d4aa; }
  .ward-info { color: #8899aa; font-size: 16px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 32px; }
  .card {
    background: #162032;
    border-radius: 16px;
    padding: 24px;
  }
  .card-label { font-size: 13px; color: #8899aa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .card-value { font-size: 48px; font-weight: 800; }
  .lp-color  { color: #00d4aa; }
  .apc-color { color: #ff6b35; }

  .discrepancy-box {
    background: #1e0a0a;
    border: 2px solid #ff4444;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 32px;
    display: none;
  }
  .discrepancy-box.visible { display: block; }
  .disc-title { color: #ff4444; font-size: 18px; font-weight: 700; margin-bottom: 16px; }
  .disc-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 15px; }
  .disc-label { color: #cc8888; }
  .disc-val { font-weight: 700; }

  .comparison {
    background: #162032;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 32px;
  }
  .comp-title { font-size: 18px; font-weight: 700; margin-bottom: 16px; }
  .comp-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid #1e3050;
  }
  .comp-row:last-child { border-bottom: none; }
  .comp-label { color: #8899aa; font-size: 14px; }
  .comp-vals { display: flex; gap: 32px; }
  .comp-val { text-align: right; }
  .comp-val span { display: block; font-size: 11px; color: #8899aa; }
  .comp-val strong { font-size: 20px; }

  .submissions {
    background: #162032;
    border-radius: 16px;
    padding: 24px;
  }
  .sub-title { font-size: 18px; font-weight: 700; margin-bottom: 16px; }
  .sub-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #1e3050;
    font-size: 14px;
    animation: fadeIn 0.4s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; } }
  .sub-item:last-child { border-bottom: none; }
  .sub-pu { color: #8899aa; }
  .sub-votes { display: flex; gap: 16px; }
  .flag-poor { color: #ffaa00; font-size: 12px; }
  .flag-bad  { color: #ff4444; font-size: 12px; }
  .counter-box {
    display: flex;
    gap: 20px;
    margin-bottom: 32px;
  }
  .counter {
    flex: 1;
    background: #162032;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
  }
  .counter-num { font-size: 48px; font-weight: 800; color: #fff; }
  .counter-label { color: #8899aa; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
  .bar-wrap { height: 10px; background: #1e3050; border-radius: 5px; margin: 12px 0 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 5px; transition: width 0.5s ease; }
  .bar-lp  { background: #00d4aa; }
  .bar-apc { background: #ff6b35; }
</style>
</head>
<body>
<header>
  <div class="logo">E Don Kast</div>
  <div class="ward-info">Ward 4 · RUMUODOMAYA · Obio-Akpor LGA · Rivers State · Live Demo</div>
</header>

<div class="counter-box">
  <div class="counter">
    <div class="counter-num" id="sub-count">0</div>
    <div class="counter-label">Forms Submitted</div>
  </div>
  <div class="counter">
    <div class="counter-num lp-color" id="lp-total">0</div>
    <div class="counter-label">LP Votes (EC8A)</div>
  </div>
  <div class="counter">
    <div class="counter-num apc-color" id="apc-total">0</div>
    <div class="counter-label">APC Votes (EC8A)</div>
  </div>
</div>

<div class="comparison">
  <div class="comp-title">EC8A Forms vs. INEC Declared — Obio-Akpor LGA</div>
  <div class="comp-row">
    <div class="comp-label">Source</div>
    <div class="comp-vals">
      <div class="comp-val"><span>LP Votes</span><strong class="lp-color">LP</strong></div>
      <div class="comp-val"><span>APC Votes</span><strong class="apc-color">APC</strong></div>
    </div>
  </div>
  <div class="comp-row">
    <div class="comp-label">📋 EC8A Forms (audience-reconstructed)</div>
    <div class="comp-vals">
      <div class="comp-val"><span>LP</span><strong class="lp-color" id="ec8a-lp">68,521</strong></div>
      <div class="comp-val"><span>APC</span><strong class="apc-color" id="ec8a-apc">16,980</strong></div>
    </div>
  </div>
  <div class="comp-row">
    <div class="comp-label">🏛 INEC Official Declared Result</div>
    <div class="comp-vals">
      <div class="comp-val"><span>LP</span><strong class="lp-color">3,829</strong></div>
      <div class="comp-val"><span>APC</span><strong class="apc-color">80,239</strong></div>
    </div>
  </div>
</div>

<div class="discrepancy-box" id="disc-box">
  <div class="disc-title">⚠ Discrepancy Detected</div>
  <div class="disc-row">
    <span class="disc-label">LP votes unaccounted for</span>
    <span class="disc-val" id="disc-lp">—</span>
  </div>
  <div class="disc-row">
    <span class="disc-label">APC votes not on EC8A forms</span>
    <span class="disc-val" id="disc-apc">—</span>
  </div>
</div>

<div class="submissions">
  <div class="sub-title">Incoming Submissions</div>
  <div id="sub-list"></div>
</div>

<script>
  const INEC_LP  = 3829;
  const INEC_APC = 80239;
  let known = 0;

  function fmt(n) { return n.toLocaleString(); }

  async function poll() {
    try {
      const res  = await fetch('/results');
      const data = await res.json();
      const results = data.results;

      // Update counters
      const total = results.length;
      let lp = 0, apc = 0;
      results.forEach(r => {
        lp  += r.lp  || 0;
        apc += r.apc || 0;
      });

      document.getElementById('sub-count').textContent = total;
      document.getElementById('lp-total').textContent  = fmt(lp);
      document.getElementById('apc-total').textContent = fmt(apc);

      // Discrepancy box — show after 3+ submissions
      if (total >= 3) {
        document.getElementById('disc-box').classList.add('visible');
        document.getElementById('disc-lp').textContent  = fmt(68521 - INEC_LP)  + ' missing';
        document.getElementById('disc-apc').textContent = fmt(INEC_APC - 16980) + ' added';
      }

      // New submissions
      if (results.length > known) {
        const list = document.getElementById('sub-list');
        const newItems = results.slice(known);
        newItems.forEach(r => {
          const div = document.createElement('div');
          div.className = 'sub-item';
          const qualColor = r.quality === 'good' ? '' :
                            r.quality === 'poor' ? 'flag-poor' : 'flag-bad';
          div.innerHTML = \`
            <div>
              <div>\${r.polling_unit || 'Unknown PU'}</div>
              <div class="sub-pu \${qualColor}">\${r.quality || ''}\${r.flags ? ' · ' + r.flags : ''}</div>
            </div>
            <div class="sub-votes">
              <span class="lp-color"><strong>\${r.lp ?? '—'}</strong> LP</span>
              <span class="apc-color"><strong>\${r.apc ?? '—'}</strong> APC</span>
            </div>\`;
          list.prepend(div);
        });
        known = results.length;
      }
    } catch(e) { console.error(e); }
    setTimeout(poll, 2000);
  }

  poll();
</script>
</body>
</html>"""


@app.route("/")
def upload_page():
    return UPLOAD_PAGE


@app.route("/dashboard")
def dashboard():
    return DASHBOARD_PAGE


@app.route("/submit", methods=["POST"])
def submit():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    if len(image_bytes) < 500:
        return jsonify({"error": "Image too small or empty"}), 400

    try:
        extraction = extract_from_image(image_bytes)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Claude returned unparseable JSON: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Store result
    parties = extraction.get("parties", {})
    flags   = extraction.get("flags", {})
    flag_list = flags.get("unreadable_fields", [])

    record = {
        "timestamp":    datetime.utcnow().isoformat(),
        "polling_unit": extraction.get("meta", {}).get("polling_unit", ""),
        "pu_code":      extraction.get("meta", {}).get("pu_code", ""),
        "lp":           parties.get("LP",  {}).get("figures"),
        "apc":          parties.get("APC", {}).get("figures"),
        "pdp":          parties.get("PDP", {}).get("figures"),
        "quality":      flags.get("quality", "unknown"),
        "flags":        ", ".join(flag_list) if flag_list else "",
        "raw":          extraction,
    }

    with _lock:
        _results.append(record)

    return jsonify(extraction)


@app.route("/results")
def results():
    with _lock:
        safe = [
            {k: v for k, v in r.items() if k != "raw"}
            for r in _results
        ]
    return jsonify({"results": safe, "count": len(safe)})


@app.route("/reset", methods=["POST"])
def reset():
    """Clear results — use before demo starts."""
    with _lock:
        _results.clear()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with:")
        print('  $env:ANTHROPIC_API_KEY = "sk-ant-..."')
        raise SystemExit(1)

    print("=" * 60)
    print("  E Don Kast — ADC 2026 Hackathon Demo")
    print("=" * 60)
    print(f"  Ward:      {WARD_NAME}")
    print(f"  LGA:       {LGA_NAME}, {STATE_NAME}")
    print()
    print("  Upload page:  http://localhost:5000/")
    print("  Dashboard:    http://localhost:5000/dashboard")
    print()
    print("  To expose publicly (ngrok):")
    print("    ngrok http 5000")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
