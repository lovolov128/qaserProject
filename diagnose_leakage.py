import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


import argparse
import joblib
import numpy as np
import pandas as pd

from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, clean_raw_data, detect_extra_columns


def main(data_path):
    print("تحميل البيانات ...")
    df = pd.read_csv(data_path)
    df = clean_raw_data(df)

    extra_numeric, extra_categorical = detect_extra_columns(df)
    print(f"\nأعمدة إضافية غير معروفة اكتُشفت: رقمية={extra_numeric}, فئوية={extra_categorical}")
    print("(لو فيه عمود هنا غريب زي 'fraud_flag' أو 'risk_label' أو شبيه، هذا على الأغلب سبب المشكلة)\n")

    # 1) ارتباط كل عمود رقمي مباشرة مع is_fraud
    print("=" * 60)
    print("ارتباط كل عمود رقمي مع is_fraud (الأقرب لـ 1.0 أو -1.0 = تسريب محتمل):")
    numeric_cols = NUMERIC_FEATURES + extra_numeric
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corrs = df[numeric_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print(corrs.to_string())

    # 2) أهمية الميزات من الموديل المدرّب فعليًا
    print("\n" + "=" * 60)
    try:
        model = joblib.load("artifacts/xgb_model.pkl")
        import json
        with open("artifacts/config.json", encoding="utf-8") as f:
            config = json.load(f)
        feature_cols = config["feature_cols"]
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        print("أهمية كل ميزة بالموديل المدرّب (لو ميزة وحدة ياخذة نسبة عالية جدًا مقارنة بالباقي = تسريب):")
        for i in order:
            print(f"  {feature_cols[i]:35s} {importances[i]:.4f}")
    except FileNotFoundError:
        print("ما لقيت artifacts/xgb_model.pkl — درّبي الموديل أول بـ train_model.py")

    # 3) فحص التكرار: هل فيه صفوف متطابقة أو شبه متطابقة تتكرر بين العملاء؟
    print("\n" + "=" * 60)
    dup_count = df.duplicated(subset=numeric_cols, keep=False).sum()
    print(f"عدد الصفوف اللي لها نفس القيم الرقمية بالضبط مع صف ثاني: {dup_count} من أصل {len(df)}")

    # 4) فحص القيم الفريدة بكل عمود فئوي — عمود بقيم قليلة جدًا مرتبطة بفئة وحدة يكشف تسريب
    print("\n" + "=" * 60)
    print("متوسط نسبة الاحتيال داخل كل فئة من كل عمود فئوي (لو فئة وحدة عندها 0% أو 100% بالضبط = تسريب):")
    for col in CATEGORICAL_FEATURES + extra_categorical:
        if col not in df.columns:
            continue
        rates = df.groupby(col)[TARGET].mean()
        extreme = rates[(rates == 0) | (rates == 1)]
        print(f"\n  {col}: {df[col].nunique()} فئة فريدة")
        if len(extreme) > 0:
            print(f"    ⚠ فئات بنسبة احتيال 0% أو 100% بالضبط: {len(extreme)} فئة")
            print(extreme.head(10).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    main(args.data)