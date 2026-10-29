import os
import sys

_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_WEB_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _WEB_DIR not in sys.path:
    sys.path.insert(0, _WEB_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from flask import Flask, jsonify, send_from_directory, request

from prediction_service import PredictionService
from prediction_service import (
    CONSTRUCTOR_COLORS,
    CONFIDENCE_BADGE_COLORS,
    DRIVER_CODE_MAP,
)

app = Flask(__name__, static_folder=None)

_service: PredictionService | None = None


def _get_service() -> PredictionService:
    global _service
    if _service is None:
        _service = PredictionService()
    return _service


@app.after_request
def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response


@app.route("/api/health", methods=["GET"])
def health():
    try:
        return jsonify({"status": "ok", "service": "F1 Predictor API"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/championship", methods=["GET"])
def championship():
    try:
        service = _get_service()
        retrain = request.args.get("retrain", "0") in ("1", "true", "yes")
        algorithm = request.args.get("algorithm", "ensemble_voting")
        result = service.get_championship_predictions(retrain=retrain, algorithm=algorithm)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config", methods=["GET"])
def config():
    try:
        return jsonify({
            "constructor_colors": CONSTRUCTOR_COLORS,
            "confidence_badge_colors": CONFIDENCE_BADGE_COLORS,
            "driver_code_map": DRIVER_CODE_MAP,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/next-race", methods=["GET"])
def next_race():
    try:
        service = _get_service()
        algorithm = request.args.get("algorithm", "ridge_regression")
        result = service.get_next_race_predictions(algorithm=algorithm)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/standings", methods=["GET"])
def standings():
    try:
        service = _get_service()
        result = service.get_current_standings()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def models():
    try:
        service = _get_service()
        algorithm = request.args.get("algorithm", None)
        result = service.get_models_info(algorithm=algorithm)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/driver/<driver_code>", methods=["GET"])
def driver_detail(driver_code):
    try:
        service = _get_service()
        result = service.get_driver_detail(driver_code)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain():
    try:
        service = _get_service()
        result = service.trigger_retrain()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    try:
        dist_index = os.path.join(_PROJECT_ROOT, "dist", "index.html")
        if os.path.exists(dist_index):
            return send_from_directory(os.path.join(_PROJECT_ROOT, "dist"), "index.html")
        splash = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>F1 2026 ML Champion Predictor</title>
<style>
  body {
    margin: 0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: linear-gradient(135deg, #0b0d17 0%, #1a1d2e 100%);
    color: #e8ecf4;
  }
  .card {
    max-width: 520px;
    padding: 40px 36px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
  }
  h1 {
    margin: 0 0 8px;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.3px;
  }
  .tag {
    display: inline-block;
    padding: 3px 10px;
    margin-bottom: 20px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 999px;
    background: #e10600;
    color: #fff;
  }
  p {
    margin: 10px 0;
    line-height: 1.6;
    color: #b5bbd0;
    font-size: 15px;
  }
  code {
    background: rgba(255, 255, 255, 0.08);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 13px;
    color: #fff;
  }
  ul {
    padding-left: 20px;
    color: #b5bbd0;
    line-height: 1.7;
  }
  .divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.08);
    margin: 20px 0;
  }
  .api {
    font-size: 13px;
    color: #8a92ab;
  }
  .api a {
    color: #6ea8ff;
    text-decoration: none;
  }
</style>
</head>
<body>
  <div class="card">
    <span class="tag">F1 2026</span>
    <h1>ML Champion Predictor</h1>
    <p>Backend API is running. The frontend build was not found.</p>
    <div class="divider"></div>
    <p><strong>To start the frontend dev server:</strong></p>
    <ul>
      <li><code>cd frontend</code></li>
      <li><code>npm install</code></li>
      <li><code>npm run dev</code></li>
    </ul>
    <p><strong>To build for production:</strong></p>
    <ul>
      <li><code>npm run build</code> (creates <code>../dist/</code>)</li>
    </ul>
    <div class="divider"></div>
    <p class="api">
      API endpoints:
      <a href="/api/health">/api/health</a> &middot;
      <a href="/api/standings">/api/standings</a> &middot;
      <a href="/api/championship">/api/championship</a> &middot;
      <a href="/api/next-race">/api/next-race</a> &middot;
      <a href="/api/models">/api/models</a>
    </p>
  </div>
</body>
</html>"""
        return splash, 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
