"""
diagnose_serving.py
يتأكد هل المشكلة بخط التقديم (serving) أو بالموديل نفسه.
ياخذ صفوف حقيقية معروف عنها is_fraud=1 و is_fraud=0 من بياناتك،
ويمررها عبر *نفس* دالة التحضير المستخدمة بـ app.py، ويطبع الاحتمالية.

تشغيل:
    python diagnose_serving.py --data bank_fraud.csv
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import argparse
import json
import joblib
import numpy as np
import pandas as pd

from features import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, CLUSTER_FEATURES, TARGET,
    build_feature_matrix, clean_raw_data,
)

ARTIFACT_DIR = "artifacts"


def main(data_path):
    print("تحميل الموديل ومكوناته...")
    model = joblib.load(f"{ARTIFACT_DIR}/xgb_model.pkl")
    kmeans = joblib.load(f"{ARTIFACT_DIR}/kmeans.pkl")
    cluster_scaler = joblib.load(f"{ARTIFACT_DIR}/cluster_scaler.pkl")
    target_encoder = joblib.load(f"{ARTIFACT_DIR}/target_encoder.pkl")
    with open(f"{ARTIFACT_DIR}/config.json", encoding="utf-8") as f:
        config = json.load(f)
    feature_cols = config["feature_cols"]
    cluster_fraud_rate = {int(k): v for k, v in config["cluster_fraud_rate"].items()}
    global_rate = config["global_fraud_rate"]

    print("تحميل البيانات وأخذ عينة من الاحتيال المعروف والآمن الحقيقي...")
    df = pd.read_csv(data_path)
    df = clean_raw_data(df)

    fraud_rows = df[df[TARGET] == 1].sample(min(10, (df[TARGET] == 1).sum()), random_state=1)
    safe_rows = df[df[TARGET] == 0].sample(10, random_state=1)

    def predict_batch(rows, label):
        cluster_input = cluster_scaler.transform(rows[CLUSTER_FEATURES])
        cluster_ids = kmeans.predict(cluster_input)
        rows = rows.copy()
        rows["cluster_risk_score"] = [cluster_fraud_rate.get(int(c), global_rate) for c in cluster_ids]

        X, _ = build_feature_matrix(rows, target_encoder, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_cols]

        proba = model.predict_proba(X)[:, 1]
        print(f"\n=== {label} (القيمة الحقيقية لـ is_fraud) ===")
        for p in sorted(proba, reverse=True):
            print(f"  احتمالية الاحتيال المتوقعة: {p:.4f}")
        print(f"  المتوسط: {proba.mean():.4f}  |  الأعلى: {proba.max():.4f}  |  الأدنى: {proba.min():.4f}")

    predict_batch(fraud_rows, "صفوف احتيال حقيقي (is_fraud=1)")
    predict_batch(safe_rows, "صفوف آمنة حقيقية (is_fraud=0)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    main(args.data)
