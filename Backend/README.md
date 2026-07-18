# قصر -

## الفكرة
- **KMeans clustering** على السلوك (المبلغ، المسافة، التوقيت...) → يستخرج "درجة خطورة العنقود"
- **XGBoost classifier** يتدرب على كل الميزات + درجة خطورة العنقود → احتمالية احتيال دقيقة
- تُقسّم النتيجة تلقائيًا لثلاث فئات بناءً على تحليل Precision/Recall حقيقي على بياناتك:
  - `safe` (آمن)
  - `mid_safe` (يتطلب حذر)
  - `fraud_danger` (خطر احتيال)

## الملفات
- `features.py` — هندسة الميزات المشتركة (تنظيف، ترميز، اختيار الأعمدة)
- `train_model.py` — التدريب الكامل، يُشغَّل مرة وحدة على بياناتك
- `app.py` — سيرفر Flask يقدّم الموديل المدرّب كـ API

## خطوة 1: التدريب )

```bash
cd backend
pip install -r requirements.txt
python train_model.py --data path/to/your_dataset.csv
```

## خطوة 2: تشغيل السيرفر

```bash
python app.py
```
يشتغل على `http://localhost:5000`

## الـ Endpoints

| Endpoint | Method | الوصف |
|---|---|---|
| `/analyze` | POST | يحلل معاملة وحدة، يرجع `tier`, `risk_score`, `top_factors` |
| `/analyze/batch` | POST | نفس الشي لعدة معاملات دفعة وحدة |
| `/metrics` | GET | تقرير دقة الموديل الكامل |
| `/model-info` | GET | الميزات المستخدمة والعتبات |
| `/health` | GET | فحص إن السيرفر شغال |

### مثال طلب `/analyze`
```json
POST /analyze
{
  "hour_of_day": 2,
  "is_weekend": 1,
  "is_night_transaction": 1,
  "country": "eg",
  "city": "cairo",
  "merchant_category": "electronics",
  "payment_method": "transfer",
  "device_type": "mobile",
  "customer_age": 30,
  "credit_score": 600,
  "account_age_years": 0.5,
  "account_balance": 500,
  "transaction_amount": 4500,
  "num_prev_transactions": 2,
  "transaction_freq_monthly": 1,
  "distance_from_home_km": 120,
  "time_since_last_txn_hrs": 400,
  "is_international": 1,
  "failed_attempts": 4,
  "pin_changed_recently": 1
}
```

### رد نموذجي
```json
{
  "tier": "fraud_danger",
  "tier_label_ar": "خطر احتيال",
  "risk_score": 87.3,
  "probability": 0.873,
  "thresholds": {"mid_safe": 0.32, "fraud_danger": 0.71},
  "top_factors": ["distance_from_home_km", "failed_attempts", "cluster_risk_score", "is_international", "transaction_amount"]
}
```
