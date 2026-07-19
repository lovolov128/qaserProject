
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from features import build_feature_matrix, clean_raw_data

app = Flask(__name__)
CORS(app)

ARTIFACT_DIR = "artifacts"

_model = joblib.load(f"{ARTIFACT_DIR}/xgb_model.pkl")
_kmeans = joblib.load(f"{ARTIFACT_DIR}/kmeans.pkl")
_cluster_scaler = joblib.load(f"{ARTIFACT_DIR}/cluster_scaler.pkl")
_target_encoder = joblib.load(f"{ARTIFACT_DIR}/target_encoder.pkl")


with open(f"{ARTIFACT_DIR}/config.json", "r", encoding="utf-8") as f:
    _config = json.load(f)

with open(f"{ARTIFACT_DIR}/metrics.json", "r", encoding="utf-8") as f:
    _metrics = json.load(f)


FEATURE_COLS = _config["feature_cols"]
NUMERIC_FEATURES = _config["numeric_features"]
CATEGORICAL_FEATURES = _config["categorical_features"]
CLUSTER_FRAUD_RATE = {int(k): v for k, v in _config["cluster_fraud_rate"].items()}
GLOBAL_FRAUD_RATE = _config["global_fraud_rate"]
MID_THRESHOLD = _config["thresholds"]["mid_safe"]
DANGER_THRESHOLD = _config["thresholds"]["fraud_danger"]

from features import CLUSTER_FEATURES 


def tier_from_probability(p: float) -> str:
    if p >= DANGER_THRESHOLD:
        return "fraud_danger"
    if p >= MID_THRESHOLD:
        return "mid_safe"
    return "safe"


TIER_LABELS_AR = {
    "safe": "آمن",
    "mid_safe": "يتطلب حذر",
    "fraud_danger": "خطر احتيال",
}


def prepare_single_row(payload: dict) -> pd.DataFrame:
    """يحوّل معاملة واحدة (JSON) لنفس شكل البيانات اللي تدرب عليها الموديل."""
    row = {col: payload.get(col) for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    df = pd.DataFrame([row])
    df = clean_raw_data(df)
    if df.empty:
        raise ValueError("قيم المعاملة غير صالحة بعد التنظيف")

    # نفس تحويل الـ clustering المستخدم وقت التدريب
    cluster_input = _cluster_scaler.transform(df[CLUSTER_FEATURES])
    cluster_id = int(_kmeans.predict(cluster_input)[0])
    df["cluster_risk_score"] = CLUSTER_FRAUD_RATE.get(cluster_id, GLOBAL_FRAUD_RATE)

    X, _ = build_feature_matrix(df, _target_encoder, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    # نضمن نفس ترتيب الأعمدة بالضبط اللي تدرب عليها الموديل
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLS]
    return X


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(force=True) or {}

    try:
        X = prepare_single_row(payload)
    except Exception as e:
        return jsonify({"error": f"تعذر معالجة بيانات المعاملة: {str(e)}"}), 400

    proba = float(_model.predict_proba(X)[0, 1])
    tier = tier_from_probability(proba)

    # أهم 5 عوامل أثرت على القرار لهذي المعاملة تحديدًا (من قيم الميزات نفسها)
    importances = _model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    top_factors = [FEATURE_COLS[i] for i in top_idx]

    return jsonify(
        {
            "tier": tier,
            "tier_label_ar": TIER_LABELS_AR[tier],
            "risk_score": round(proba * 100, 1),
            "probability": round(proba, 4),
            "thresholds": {"mid_safe": MID_THRESHOLD, "fraud_danger": DANGER_THRESHOLD},
            "top_factors": top_factors,
        }
    )


@app.route("/analyze/batch", methods=["POST"])
def analyze_batch():
    """نفس /analyze بس لعدة معاملات دفعة وحدة: {"transactions": [ {...}, {...} ]}"""
    payload = request.get_json(force=True) or {}
    transactions = payload.get("transactions", [])
    if not transactions:
        return jsonify({"error": "لا توجد معاملات بالطلب"}), 400

    results = []
    for tx in transactions:
        try:
            X = prepare_single_row(tx)
            proba = float(_model.predict_proba(X)[0, 1])
            tier = tier_from_probability(proba)
            results.append(
                {
                    "tier": tier,
                    "tier_label_ar": TIER_LABELS_AR[tier],
                    "risk_score": round(proba * 100, 1),
                }
            )
        except Exception as e:
            results.append({"error": str(e)})

    return jsonify({"results": results})


@app.route("/metrics", methods=["GET"])
def metrics():
    """تقرير دقة الموديل (يُستخدم لعرض شارة/لوحة أداء بالفرونت)."""
    return jsonify(_metrics)


@app.route("/model-info", methods=["GET"])
def model_info():
    """معلومات عن الموديل: الميزات المستخدمة والعتبات، مفيد للتوثيق بالفرونت."""
    return jsonify(
        {
            "feature_count": len(FEATURE_COLS),
            "features": FEATURE_COLS,
            "thresholds": _config["thresholds"],
            "cluster_fraud_rates": _config["cluster_fraud_rate"],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
