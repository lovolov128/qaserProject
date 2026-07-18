"""
features.py
منطق هندسة الميزات المشترك بين التدريب (train_model.py) والتقديم (app.py)
عشان نضمن إن نفس التحويلات تنطبق بالظبط وقت التدريب ووقت أي طلب تحقق حقيقي.
"""

import numpy as np
import pandas as pd

# الأعمدة الرقمية المعروفة من الـ dataset (سلوك + بيانات عميل)
NUMERIC_FEATURES = [
    "hour_of_day",
    "is_weekend",
    "is_night_transaction",
    "customer_age",
    "credit_score",
    "account_age_years",
    "account_balance",
    "transaction_amount",
    "num_prev_transactions",
    "transaction_freq_monthly",
    "distance_from_home_km",
    "time_since_last_txn_hrs",
    "is_international",
    "failed_attempts",
    "pin_changed_recently",
]

# الأعمدة الفئوية (categorical) — هذي أعلى خطورة تحتاج ترميز خاص (target encoding)
CATEGORICAL_FEATURES = [
    "country",
    "city",
    "merchant_category",
    "payment_method",
    "device_type",
]

# الميزات السلوكية اللي نستخدمها للـ clustering فقط (سلوك المعاملة نفسها،
# مو بيانات العميل الثابتة زي العمر أو الدرجة الائتمانية)
CLUSTER_FEATURES = [
    "transaction_amount",
    "distance_from_home_km",
    "time_since_last_txn_hrs",
    "hour_of_day",
    "is_night_transaction",
    "num_prev_transactions",
    "transaction_freq_monthly",
    "failed_attempts",
]

ID_DATE_COLUMNS = ["transaction_id", "customer_id", "transaction_date", "transaction_time"]
TARGET = "is_fraud"

# أعمدة "بعد الحدث" (post-outcome) — معلومات ما تكون متوفرة وقت المعاملة الفعلية
# لأنها أصلاً نتيجة لاحقة للاحتيال، مو سبب له. تدريب الموديل عليها = تسريب بيانات (data leakage).
# fraud_type مثال واضح: يسجل نوع الاحتيال بعد ما يصير، فهو عمليًا نسخة ثانية من is_fraud.
LEAKY_COLUMNS = ["fraud_type", "fraud_reason", "fraud_label", "risk_flag", "risk_label", "fraud_category"]


def detect_extra_columns(df: pd.DataFrame) -> list:
    """
    يكتشف أي أعمدة زيادة موجودة بالداتاست ما ذكرناها بالقوائم أعلاه
    (احتياط للعمود الـ26 المجهول أو أي عمود إضافي بالنسخة الحقيقية).
    يضيفها تلقائيًا كرقمية أو فئوية حسب نوعها.
    """
    known = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ID_DATE_COLUMNS + LEAKY_COLUMNS + [TARGET])
    extra_numeric, extra_categorical = [], []
    for col in df.columns:
        if col in known:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            extra_numeric.append(col)
        else:
            extra_categorical.append(col)
    return extra_numeric, extra_categorical


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """تنظيف أولي: تكرارات، قيم مفقودة، قيم غير منطقية."""
    df = df.drop_duplicates()

    # نتأكد الأعمدة الرقمية المتوقعة فعليًا رقمية (لو انقرت كنص بالخطأ)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # تعبئة القيم المفقودة: median للرقمي، "unknown" للفئوي
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

    # قيم غير منطقية: مبالغ سالبة، أعمار غير واقعية
    if "transaction_amount" in df.columns:
        df = df[df["transaction_amount"] >= 0]
    if "customer_age" in df.columns:
        df = df[(df["customer_age"] >= 15) & (df["customer_age"] <= 100)]
    if "account_balance" in df.columns:
        df["account_balance"] = df["account_balance"].clip(lower=0)

    df = df.reset_index(drop=True)
    return df


class TargetEncoder:
    """
    ترميز الأعمدة الفئوية بمتوسط نسبة الاحتيال التاريخية لكل فئة
    (مثلاً: كل دولة/مدينة/طريقة دفع تاخذ "درجة خطورة" بدل رقم عشوائي).
    يُدرَّب فقط على بيانات التدريب لتفادي تسريب المعلومات (data leakage).
    smoothing يمنع الفئات النادرة من أخذ قيمة متطرفة غير موثوقة.
    """

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing
        self.mappings = {}
        self.global_mean = 0.5

    def fit(self, df: pd.DataFrame, target: pd.Series, columns: list):
        self.global_mean = target.mean()
        for col in columns:
            if col not in df.columns:
                continue
            stats = target.groupby(df[col]).agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + self.global_mean * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self.mappings[col] = smoothed.to_dict()
        return self

    def transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            mapping = self.mappings.get(col, {})
            df[f"{col}_risk"] = df[col].map(mapping).fillna(self.global_mean)
        return df


def build_feature_matrix(df, target_encoder, categorical_cols, numeric_cols, extra_numeric=None):
    """يبني مصفوفة الميزات النهائية (رقمي + فئوي مُرمّز) الجاهزة للموديل."""
    extra_numeric = extra_numeric or []
    df_enc = target_encoder.transform(df, categorical_cols)
    risk_cols = [f"{c}_risk" for c in categorical_cols if f"{c}_risk" in df_enc.columns]
    feature_cols = numeric_cols + extra_numeric + risk_cols
    if "cluster_risk_score" in df_enc.columns:
        feature_cols.append("cluster_risk_score")
    feature_cols = [c for c in feature_cols if c in df_enc.columns]
    return df_enc[feature_cols], feature_cols