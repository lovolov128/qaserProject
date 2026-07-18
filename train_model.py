import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import (
    CATEGORICAL_FEATURES,
    CLUSTER_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    TargetEncoder,
    build_feature_matrix,
    clean_raw_data,
    detect_extra_columns,
)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    print(f"[1/6] تحميل البيانات من {path} ...")
    df = pd.read_csv(path)
    print(f"      الحجم الخام: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] تنظيف البيانات ...")
    before = len(df)
    df = clean_raw_data(df)
    print(f"      بعد التنظيف: {df.shape} (حُذف {before - len(df)} صف)")
    return df


def run_clustering(train_df, test_df, n_clusters=5):
    """
    Clustering على الميزات السلوكية فقط، ثم نحسب نسبة الاحتيال الفعلية
    داخل كل عنقود من بيانات التدريب -> تتحول لميزة "cluster_risk_score"
    تُغذّى لموديل الـ classification. هذا يمسك أنماط شاذة قد يفوتها classifier
    وحده لو الحالة نادرة أو مختلفة عن الأمثلة المعروفة.
    """
    print(f"[3/6] Clustering سلوكي (KMeans, k={n_clusters}) ...")
    scaler = StandardScaler()
    X_train_cluster = scaler.fit_transform(train_df[CLUSTER_FEATURES])
    X_test_cluster = scaler.transform(test_df[CLUSTER_FEATURES])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["cluster"] = kmeans.fit_predict(X_train_cluster)
    test_df["cluster"] = kmeans.predict(X_test_cluster)

    cluster_fraud_rate = train_df.groupby("cluster")[TARGET].mean().to_dict()
    global_rate = train_df[TARGET].mean()

    train_df["cluster_risk_score"] = train_df["cluster"].map(cluster_fraud_rate)
    test_df["cluster_risk_score"] = test_df["cluster"].map(cluster_fraud_rate).fillna(global_rate)

    print("      نسبة الاحتيال داخل كل عنقود (train):")
    for c, rate in sorted(cluster_fraud_rate.items()):
        print(f"        عنقود {c}: {rate:.2%}")

    return train_df, test_df, scaler, kmeans, cluster_fraud_rate, global_rate


def train_classifier(X_train, y_train):
    print("[4/6] تدريب XGBoost Classifier ...")
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"      توازن الفئات -> آمن: {neg}, احتيال: {pos} (scale_pos_weight={scale_pos_weight:.2f})")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, feature_cols):
    print("[5/6] تقييم الدقة ...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, target_names=["not_fraud", "fraud"], output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"      ROC-AUC: {roc_auc:.4f}   |   PR-AUC: {pr_auc:.4f}")
    print(f"      Precision(fraud): {report['fraud']['precision']:.3f}   Recall(fraud): {report['fraud']['recall']:.3f}   F1(fraud): {report['fraud']['f1-score']:.3f}")

    # --- رسومات ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["آمن", "احتيال"]).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/confusion_matrix.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/roc_curve.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
    plt.title(f"Precision-Recall Curve (AP={pr_auc:.3f})")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/precision_recall_curve.png", dpi=140)
    plt.close()

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([feature_cols[i] for i in order][::-1], importances[order][::-1], color="#3B5FC4")
    plt.title("أهم 15 ميزة أثّرت على تقييم الاحتيال")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/feature_importance.png", dpi=140)
    plt.close()

    metrics = {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision_fraud": round(float(report["fraud"]["precision"]), 4),
        "recall_fraud": round(float(report["fraud"]["recall"]), 4),
        "f1_fraud": round(float(report["fraud"]["f1-score"]), 4),
        "accuracy": round(float(report["accuracy"]), 4),
        "confusion_matrix": cm.tolist(),
        "n_test_samples": int(len(y_test)),
    }
    return metrics, y_proba


def find_thresholds(y_test, y_proba, target_precision=0.20, target_recall=0.70):
    """
    يحدد عتبتين لتقسيم النتيجة لثلاث فئات، بالاعتماد على منحنى precision-recall
    الفعلي بدل أرقام عشوائية:
      - عتبة "Fraud Danger": أصغر عتبة تحقق precision >= target_precision
        (يعني لما نقول "احتيال" نطمن إنها غالبًا صح، نتفادى إزعاج المستخدمين
        بإنذارات كاذبة كثيرة).
      - عتبة "Mid Safe": أصغر عتبة تمسك على الأقل target_recall من كل حالات
        الاحتيال الحقيقية ضمن فئتي (Mid + Danger) مجتمعتين — يعني قلة قليلة
        فقط من الاحتيال الحقيقي تفوتنا للفئة الآمنة.
    """
    thresholds = np.linspace(0.05, 0.95, 19)

    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        rows.append((t, precision, recall, f1))

    # عتبة الخطورة العالية: أول عتبة (من الأعلى للأسفل) تحقق الدقة المطلوبة.
    # لو ما تحققت بأي عتبة (يعني الهدف أعلى من قدرة الموديل الفعلية)، نرجع
    # للعتبة اللي تعطي أعلى precision ممكنة فعليًا بدل رقم افتراضي عشوائي.
    danger_threshold = None
    for t, precision, recall, f1 in sorted(rows, key=lambda r: -r[0]):
        if precision >= target_precision:
            danger_threshold = t
            break
    if danger_threshold is None:
        danger_threshold = max(rows, key=lambda r: r[1])[0]
        print(f"      ⚠ ما وصلنا precision >= {target_precision} بأي عتبة، استخدمنا أعلى precision فعلي متاح")

    # عتبة المتوسط: أكبر عتبة ممكنة لسا تحافظ على نسبة استرجاع كافية
    # (نبيها أكبر ما يمكن عشان فئة "آمن" تبقى ذات معنى وما تصير كل شي "يتطلب حذر").
    # الاسترجاع يقل كل ما زادت العتبة، فنبحث من الأعلى للأسفل عن أول عتبة تحقق الهدف.
    mid_threshold = None
    for t, precision, recall, f1 in sorted(rows, key=lambda r: -r[0]):
        if recall >= target_recall:
            mid_threshold = t
            break
    if mid_threshold is None:
        mid_threshold = min(rows, key=lambda r: r[0])[0]
        print(f"      ⚠ ما وصلنا recall >= {target_recall} بأي عتبة، استخدمنا أصغر عتبة مجربة")

    if mid_threshold >= danger_threshold:
        mid_threshold = max(0.05, danger_threshold - 0.15)

    print("[6/6] تحديد عتبات التصنيف الثلاثي:")
    print(f"      Mid Safe يبدأ من احتمالية >= {mid_threshold:.2f}")
    print(f"      Fraud Danger يبدأ من احتمالية >= {danger_threshold:.2f}")
    print("\n      جدول Precision/Recall/F1 الكامل حسب العتبة (للمراجعة اليدوية إذا احتجتِ تعدّلين العتبات يدويًا):")
    print("      threshold | precision | recall  |  f1")
    for t, precision, recall, f1 in rows:
        print(f"        {t:.2f}    |   {precision:.3f}   | {recall:.3f}  | {f1:.3f}")

    return float(mid_threshold), float(danger_threshold)


def main(data_path, target_precision=0.20, target_recall=0.70):
    df = load_data(data_path)
    df = clean_data(df)

    extra_numeric, extra_categorical = detect_extra_columns(df)
    if extra_numeric or extra_categorical:
        print(f"      ⚠ أعمدة إضافية غير معروفة اتلقيناها تلقائيًا -> رقمية: {extra_numeric}, فئوية: {extra_categorical}")

    all_categorical = CATEGORICAL_FEATURES + extra_categorical
    all_numeric = NUMERIC_FEATURES + extra_numeric

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )

    train_df, test_df, cluster_scaler, kmeans, cluster_fraud_rate, global_rate = run_clustering(
        train_df, test_df
    )

    target_encoder = TargetEncoder(smoothing=20.0)
    target_encoder.fit(train_df, train_df[TARGET], all_categorical)

    X_train, feature_cols = build_feature_matrix(train_df, target_encoder, all_categorical, all_numeric)
    X_test, _ = build_feature_matrix(test_df, target_encoder, all_categorical, all_numeric)
    y_train, y_test = train_df[TARGET], test_df[TARGET]

    print(f"      عدد الميزات النهائية المستخدمة بالتدريب: {len(feature_cols)}")
    print(f"      الميزات: {feature_cols}")

    model = train_classifier(X_train, y_train)
    metrics, y_proba = evaluate(model, X_test, y_test, feature_cols)
    mid_threshold, danger_threshold = find_thresholds(
        y_test.values, y_proba, target_precision=target_precision, target_recall=target_recall
    )

    # --- حفظ كل شي يحتاجه الـ API لاحقًا ---
    joblib.dump(model, f"{ARTIFACT_DIR}/xgb_model.pkl")
    joblib.dump(kmeans, f"{ARTIFACT_DIR}/kmeans.pkl")
    joblib.dump(cluster_scaler, f"{ARTIFACT_DIR}/cluster_scaler.pkl")
    joblib.dump(target_encoder, f"{ARTIFACT_DIR}/target_encoder.pkl")

    config = {
        "feature_cols": feature_cols,
        "numeric_features": all_numeric,
        "categorical_features": all_categorical,
        "cluster_fraud_rate": {str(k): v for k, v in cluster_fraud_rate.items()},
        "global_fraud_rate": global_rate,
        "thresholds": {"mid_safe": mid_threshold, "fraud_danger": danger_threshold},
    }
    with open(f"{ARTIFACT_DIR}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(f"{ARTIFACT_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n✅ خلص التدريب. كل الملفات محفوظة بمجلد artifacts/:")
    print("   - xgb_model.pkl, kmeans.pkl, cluster_scaler.pkl, target_encoder.pkl")
    print("   - config.json (الميزات والعتبات)")
    print("   - metrics.json (تقرير الدقة)")
    print("   - confusion_matrix.png, roc_curve.png, precision_recall_curve.png, feature_importance.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="مسار ملف الـ CSV")
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.20,
        help="أدنى precision مقبولة لفئة Fraud Danger (افتراضي 0.20)",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.70,
        help="أدنى recall مقبولة لفئتي Mid+Danger مجتمعتين (افتراضي 0.70)",
    )
    args = parser.parse_args()
    main(args.data, target_precision=args.target_precision, target_recall=args.target_recall)