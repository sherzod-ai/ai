# Preprocessing va Encoding AI Model Aniqligiga Ta'siri

**Magistrlik dissertatsiyasi himoyasi uchun amaliy AI ilovasi**

Muallif: **Tursunov Sherzod Abduvakil ugli** · Sharof Rashidov nomidagi SamDU magistranti · Sun'iy intellekt yo'nalishi

---

## Maqsad

Foydalanuvchi CSV dataset yuklaydi. Ilova datasetni dastlabki statistik tahlil qiladi, preprocessing bajaradi, kategoriyali ustunlarga turli encoding usullarini qo'llaydi, klassifikatsiya modelini o'qitadi va **encoding usullari model aniqligiga qanday ta'sir qilishini metrikalar orqali taqqoslab beradi**.

Bu ilova **IBM Telco Customer Churn** dataseti misolida sinab ko'rilgan, lekin har qanday CSV dataset bilan ishlay oladi.

---

## O'rnatish

```bash
# 1) Kerakli kutubxonalarni o'rnatish
pip install -r requirements.txt

# 2) Ishga tushirish
streamlit run app.py
```

Brauzer avtomatik ochiladi (`http://localhost:8501`).

---

## Ishlatilgan kutubxonalar

| Kutubxona | Maqsad |
|---|---|
| `streamlit` | Web interfeys |
| `pandas`, `numpy` | Ma'lumotlar bilan ishlash |
| `scikit-learn` | ML modellar, scaling, encoding, GridSearchCV |
| `plotly` | Interaktiv grafiklar (bar, box, heatmap, histogram) |
| `matplotlib` | Grafiklar (zaxira) |
| `joblib` | Modelni saqlash/yuklash |

---

## Foydalanish yo'riqnomasi (qisqacha)

Yon paneldagi 11 bosqichni ketma-ket bajaring:

1. **Dataset yuklash** — CSV faylni yuklang, target ustunni tanlang
2. **EDA** — statistik tahlil, disbalans tekshirish, grafiklar
3. **Data Cleaning** — duplicates, whitespace, hidden NaN, numeric conversion
4. **Missing Values** — drop / mean / median / mode / avtomatik
5. **Outlier tahlili** — IQR usuli, boxplot, qoldirish / olib tashlash / cap
6. **Feature Scaling** — None / StandardScaler / MinMaxScaler
7. **Encoding usullari** — Label / One-Hot / Frequency / Target encoding (4 ta usul)
8. **Model o'qitish** — Random Forest (default) yoki 4 boshqa model, ixtiyoriy GridSearchCV
9. **Modelni baholash** — Accuracy, Precision, Recall, F1, Confusion Matrix, Feature Importance
10. **Encoding taqqoslash** — 4 ta encoding ni bir xil splitda taqqoslash, bar chart, eng yaxshi tanlov
11. **Yuklab olish** — preprocessed CSV, encoded CSV, trained model (.joblib), report.txt

---

## Telco Customer Churn datasetida test qilish

Dataset bepul yuklab olinadi (~177 KB):

- **Kaggle:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **GitHub (to'g'ridan-to'g'ri):** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

**Tavsiya etilgan ish ketma-ketligi:**

1. Yuklang `Telco-Customer-Churn.csv` faylni → 1-bo'lim
2. Target sifatida **`Churn`** ni tanlang → 1-bo'lim
3. 2-bo'limda EDA — disbalans haqida ogohlantirish chiqadi (73.5% / 26.5%)
4. 3-bo'limda **TotalCharges** ni numeric ga aylantiring (tugma mavjud)
5. 4-bo'limda **"Avtomatik"** tanlov bilan missing values to'ldiring (11 ta NaN)
6. 5-bo'limda outlier qoldiring (Random Forest sezgir emas)
7. 6-bo'limda scaling shart emas (Random Forest uchun)
8. 7-bo'limda har qaysi encoding ni alohida sinab ko'ring
9. 8-bo'limda Random Forest tanlang, ixtiyoriy GridSearchCV yoqing
10. 9-bo'limda F1-score va Confusion Matrix ni tahlil qiling
11. **10-bo'limda** — eng muhim qism — 4 ta encoding ni avtomatik taqqoslang
12. 11-bo'limda barcha natijalarni yuklab oling

---

## Texnik tafsilotlar

### Target Encoding leakage'siz implementatsiya

Target Encoding kuchli, lekin xavfli — agar to'g'ri qilinmasa, model test javobini "ko'rib qoladi" va soxta yuqori natija beradi.

**Bu ilovadagi yondashuv:**
- **Train ichida:** K-Fold (5-Fold) out-of-fold encoding — har qator uchun encoding qiymati u BO'LMAGAN foldlardan hisoblanadi
- **Val/Test uchun:** butun train asosida yakuniy mapping (smoothing bilan)
- **Test target hech qachon ko'rinmaydi** — `target_encode_splits()` funksiyasiga `y_test` umuman yuborilmaydi

### Split nisbati

70% train · 15% validation · 15% test · stratified · `random_state=42`

### GridSearchCV parametrlari

```python
n_estimators: [100, 200, 300]
max_depth: [5, 10, 15, None]
min_samples_split: [2, 5, 10]
max_features: ["sqrt", "log2"]
scoring: "f1_weighted"
cv: 5-fold
```

---

## Fayl tuzilmasi

```
streamlit-app/
├── app.py              # Asosiy ilova (barcha 11 bo'lim)
├── requirements.txt    # Python kutubxonalari
└── README.md           # Bu fayl
```

---

## Litsenziya va foydalanish

Magistrlik dissertatsiyasi tarkibida ishlatish uchun mo'ljallangan. Akademik foydalanish uchun ochiq.

**Aloqa:** sherzodtursunov943@gmail.com · +998 90 605 17 75
