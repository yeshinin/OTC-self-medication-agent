"""
OTC Self-Medication Agent — Flask API
======================================
Minimal server that exposes the pipeline as a REST endpoint.

Usage:
    pip install flask flask-cors
    PYTHONPATH=src python app.py

Endpoints:
    POST /analyze   { "query": "I take NyQuil and Tylenol" }
                 →  { risk_level, risk_summary, details, what_to_do,
                      see_pharmacist, sources, extracted }

    GET  /health  →  { "status": "ok" }
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from synthesis_layer import run_from_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
CORS(app)


'''@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")'''

@app.route("/")
def index():
    print(f"Looking for index.html in: {BASE_DIR}")
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)

    if not data or not data.get("query", "").strip():
        return jsonify({"error": "No query provided"}), 400

    query  = data["query"].strip()
    report = run_from_text(query)

    if report.get("error"):
        return jsonify({"error": report["error"]}), 500

    output = report.get("output", {})

    return jsonify({
        "risk_level":    output.get("risk_level", "none"),
        "risk_summary":  output.get("risk_summary", ""),
        "details":       output.get("details", []),
        "what_to_do":    output.get("what_to_do", ""),
        "see_pharmacist": output.get("see_pharmacist", False),
        "sources":       report.get("sources", []),
        "extracted": {
            "products":   [p["name"] for p in
                           report.get("extracted", {}).get("products", [])],
            "substances": [s["name"] for s in
                           report.get("extracted", {}).get("substances", [])],
            "notes":      report.get("extracted", {})
                               .get("extraction_notes", ""),
        }
    })


if __name__ == "__main__":
    print("\n  OTC Agent running — open http://localhost:5000 in your browser.\n")
    app.run(debug=True, port=5000)
