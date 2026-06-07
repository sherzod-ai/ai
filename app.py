"""
Preprocessing va Encoding AI Model Aniqligiga Ta'siri
=====================================================
Streamlit ilovasi — Tursunov Sherzod Abduvakil ugli (SamDU magistranti).

Magistrlik dissertatsiyasi himoyasida ko'rsatish uchun amaliy AI loyihasi.
Foydalanuvchi CSV dataset yuklaydi, EDA va preprocessing bajariladi,
4 xil encoding usuli model aniqligiga ta'sirini taqqoslab beradi.
"""

import io
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# =====================================================================
# PAGE CONFIG VA STIL
# =====================================================================
st.set_page_config(
    page_title="Preprocessing va Encoding AI Model Aniqligiga Ta'siri",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1F3864"   # to'q ko'k
ACCENT = "#D97706"    # to'q sariq (amber)
SUCCESS = "#10B981"
DANGER = "#DC2626"
MUTED = "#64748B"
BG_TINT = "#F1F5F9"

st.markdown(f"""
<style>
  .main {{ background-color: #FFFFFF; }}
  .stApp {{ background-color: #FFFFFF; }}
  section[data-testid="stSidebar"] {{ background-color: {BG_TINT}; }}
  h1, h2, h3 {{ color: {PRIMARY}; }}
  .stRadio > label {{ color: {PRIMARY}; font-weight: 600; }}
  div[data-testid="stMetricValue"] {{ color: {PRIMARY}; }}
  .stButton > button {{
    background-color: {PRIMARY}; color: white; border: 0;
    border-radius: 6px; font-weight: 600;
  }}
  .stButton > button:hover {{ background-color: {ACCENT}; }}
  .info-box {{
    background-color: {BG_TINT}; border-left: 4px solid {ACCENT};
    padding: 12px 16px; border-radius: 4px; margin: 10px 0;
  }}
  .success-box {{
    background-color: #D1FAE5; border-left: 4px solid {SUCCESS};
    padding: 12px 16px; border-radius: 4px; margin: 10px 0;
  }}
  .warning-box {{
    background-color: #FEF3E2; border-left: 4px solid {ACCENT};
    padding: 12px 16px; border-radius: 4px; margin: 10px 0;
  }}
</style>
""", unsafe_allow_html=True)


def info(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def success_box(text: str):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def warn_box(text: str):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)


# =====================================================================
# SESSION STATE INIT
# =====================================================================
def init_state():
    defaults = {
        "df_original": None,
        "df_current": None,
        "target_col": None,
        "numeric_cols": [],
        "categorical_cols": [],
        "model_results": {},
        "trained_model": None,
        "best_encoding": None,
        "best_params": None,
        "feature_importance_df": None,
        "encoded_df": None,
        "comparison_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# =====================================================================
# HELPER: detect column types
# =====================================================================
def detect_column_types(df: pd.DataFrame):
    """
    Sonli va kategoriyali ustunlarni aniqlaydi.
    MUHIM: Faqat 2 ta noyob qiymatga ega sonli ustunlar (masalan SeniorCitizen 0/1)
    avtomatik ravishda kategoriyali sifatida belgilanadi.
    """
    raw_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = []
    for col in raw_numeric:
        if df[col].nunique() <= 2:
            categorical_cols.append(col)  # binary flag (0/1) -> kategoriyali
        else:
            numeric_cols.append(col)
    return numeric_cols, categorical_cols


# =====================================================================
# ENCODING IMPLEMENTATIONS — LEAKAGE'SIZ
# =====================================================================
def label_encode_splits(X_train, X_val, X_test, cat_cols):
    """Label encoding — har ustun uchun mapping train da fit qilinadi."""
    X_train_e = X_train.copy()
    X_val_e = X_val.copy() if X_val is not None else None
    X_test_e = X_test.copy() if X_test is not None else None
    for col in cat_cols:
        # Train asosida mapping yaratamiz
        train_vals = X_train_e[col].astype(str)
        unique_vals = train_vals.unique().tolist()
        mapping = {v: i for i, v in enumerate(unique_vals)}
        # Yangi (ko'rilmagan) kategoriyalar uchun fallback qiymat
        fallback = -1
        X_train_e[col] = train_vals.map(mapping).fillna(fallback).astype(int)
        if X_val_e is not None:
            X_val_e[col] = X_val_e[col].astype(str).map(mapping).fillna(fallback).astype(int)
        if X_test_e is not None:
            X_test_e[col] = X_test_e[col].astype(str).map(mapping).fillna(fallback).astype(int)
    return X_train_e, X_val_e, X_test_e


def onehot_encode_splits(X_train, X_val, X_test, cat_cols):
    """One-Hot encoding — train ustunlari asosida val/test ham bir xil ustunlarga keltiriladi."""
    X_train_e = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
    X_train_e = X_train_e.astype({c: int for c in X_train_e.columns if X_train_e[c].dtype == bool})
    if X_val is not None:
        X_val_e = pd.get_dummies(X_val, columns=cat_cols, drop_first=False)
        X_val_e = X_val_e.reindex(columns=X_train_e.columns, fill_value=0)
    else:
        X_val_e = None
    if X_test is not None:
        X_test_e = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
        X_test_e = X_test_e.reindex(columns=X_train_e.columns, fill_value=0)
    else:
        X_test_e = None
    return X_train_e, X_val_e, X_test_e


def frequency_encode_splits(X_train, X_val, X_test, cat_cols):
    """Frequency encoding — chastota mapping train asosida hisoblanadi."""
    X_train_e = X_train.copy()
    X_val_e = X_val.copy() if X_val is not None else None
    X_test_e = X_test.copy() if X_test is not None else None
    for col in cat_cols:
        freq_map = X_train_e[col].astype(str).value_counts(normalize=True).to_dict()
        X_train_e[col] = X_train_e[col].astype(str).map(freq_map).fillna(0.0).astype(float)
        if X_val_e is not None:
            X_val_e[col] = X_val_e[col].astype(str).map(freq_map).fillna(0.0).astype(float)
        if X_test_e is not None:
            X_test_e[col] = X_test_e[col].astype(str).map(freq_map).fillna(0.0).astype(float)
    return X_train_e, X_val_e, X_test_e


def target_encode_splits(X_train, y_train, X_val, X_test, cat_cols, n_folds=5, smoothing=10):
    """
    K-Fold Target Encoding — DATA LEAKAGE OLDINI OLISH UCHUN.

    MUHIM IZOH: Target Encoding test target qiymatlarini KO'RMASLIGI KERAK.
    - Train ning har bir qatori uchun encoding qiymati u BO'LMAGAN foldlardan hisoblanadi
      (out-of-fold). Bu — train ichidagi leakage'ni oldini oladi.
    - Val va test uchun yakuniy mapping butun train asosida hisoblanadi.
    - Smoothing: kam namunali kategoriyalar uchun global mean bilan aralashtirish.
    """
    X_train_e = X_train.copy()
    X_val_e = X_val.copy() if X_val is not None else None
    X_test_e = X_test.copy() if X_test is not None else None

    # Xavfsizlik: y_train string bo'lsa (masalan "Yes"/"No") sonlashtiramiz
    # Pandas 2.x StringDtype yoki object dtype bo'lsa ham ishlaydi
    y_train = pd.Series(y_train).reset_index(drop=True)
    if not pd.api.types.is_numeric_dtype(y_train):
        y_train, _ = pd.factorize(y_train)
        y_train = pd.Series(y_train)

    X_train_idx_reset = X_train.reset_index(drop=True)
    global_mean = float(y_train.mean())

    for col in cat_cols:
        # 1) Train uchun out-of-fold encoding
        oof_values = pd.Series(index=X_train_idx_reset.index, dtype=float)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X_train_idx_reset):
            fold_train_col = X_train_idx_reset.iloc[tr_idx][col].astype(str)
            fold_train_y = y_train.iloc[tr_idx]
            # Smoothed mean per category
            agg = fold_train_y.groupby(fold_train_col).agg(["mean", "count"])
            smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
            mapping = smooth.to_dict()
            fold_val_col = X_train_idx_reset.iloc[val_idx][col].astype(str)
            oof_values.iloc[val_idx] = fold_val_col.map(mapping).fillna(global_mean).values

        X_train_e[col] = oof_values.values

        # 2) Val/test uchun butun train asosida yakuniy mapping
        full_col = X_train_idx_reset[col].astype(str)
        agg_full = y_train.groupby(full_col).agg(["mean", "count"])
        smooth_full = (agg_full["count"] * agg_full["mean"] + smoothing * global_mean) / (agg_full["count"] + smoothing)
        final_mapping = smooth_full.to_dict()

        if X_val_e is not None:
            X_val_e[col] = X_val_e[col].astype(str).map(final_mapping).fillna(global_mean).astype(float)
        if X_test_e is not None:
            X_test_e[col] = X_test_e[col].astype(str).map(final_mapping).fillna(global_mean).astype(float)

    return X_train_e, X_val_e, X_test_e


# =====================================================================
# HELPER: split + encode + train + evaluate (taqqoslash uchun)
# =====================================================================
def get_classifier(name: str, random_state: int = 42):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, max_depth=10,
                                       min_samples_split=5, random_state=random_state, n_jobs=-1)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=random_state)
    if name == "KNN":
        return KNeighborsClassifier()
    if name == "Naive Bayes":
        return GaussianNB()
    return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)


def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "y_pred": y_pred,
    }


def run_encoding_pipeline(df, target_col, cat_cols, num_cols, encoding_method,
                          model_name="Random Forest", apply_scaling=False,
                          test_size=0.15, val_size=0.15, random_state=42):
    """To'liq pipeline: split → encode → (scale) → train → evaluate. Tahlil uchun."""
    X = df[num_cols + cat_cols].copy()
    y = df[target_col].copy()

    # Target encode label uchun
    # Barcha string dtype larni handle qilamiz (object, StringDtype, category)
    if not pd.api.types.is_numeric_dtype(y):
        y, _ = pd.factorize(y)
        y = pd.Series(y)

    # Stratified split: avval test, keyin train+val
    stratify = y if len(np.unique(y)) > 1 else None
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    stratify_tv = y_tv if len(np.unique(y_tv)) > 1 else None
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_rel, random_state=random_state, stratify=stratify_tv
    )

    # Encoding
    if encoding_method == "Label Encoding":
        Xtr_e, Xv_e, Xte_e = label_encode_splits(X_train, X_val, X_test, cat_cols)
    elif encoding_method == "One-Hot Encoding":
        Xtr_e, Xv_e, Xte_e = onehot_encode_splits(X_train, X_val, X_test, cat_cols)
    elif encoding_method == "Frequency Encoding":
        Xtr_e, Xv_e, Xte_e = frequency_encode_splits(X_train, X_val, X_test, cat_cols)
    elif encoding_method == "Target Encoding":
        Xtr_e, Xv_e, Xte_e = target_encode_splits(X_train, pd.Series(y_train), X_val, X_test, cat_cols)
    else:
        raise ValueError(f"Noma'lum encoding usuli: {encoding_method}")

    # Scaling (sonli ustunlarga, train asosida)
    if apply_scaling:
        scaler = StandardScaler()
        num_cols_present = [c for c in num_cols if c in Xtr_e.columns]
        if num_cols_present:
            Xtr_e[num_cols_present] = scaler.fit_transform(Xtr_e[num_cols_present])
            Xv_e[num_cols_present] = scaler.transform(Xv_e[num_cols_present])
            Xte_e[num_cols_present] = scaler.transform(Xte_e[num_cols_present])

    # Train
    model = get_classifier(model_name, random_state)
    model.fit(Xtr_e, y_train)

    # Evaluate on test
    test_metrics = evaluate_classifier(model, Xte_e, y_test)
    val_metrics = evaluate_classifier(model, Xv_e, y_val)

    return {
        "model": model,
        "X_train": Xtr_e, "X_val": Xv_e, "X_test": Xte_e,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "feature_columns": Xtr_e.columns.tolist(),
    }


# =====================================================================
# SIDEBAR — NAVIGATSIYA
# =====================================================================
st.sidebar.markdown(f"<h2 style='color:{PRIMARY};margin-bottom:0'>Bo'limlar</h2>", unsafe_allow_html=True)
st.sidebar.markdown(f"<p style='color:{MUTED};font-size:12px'>Tursunov Sherzod · SamDU 2026</p>", unsafe_allow_html=True)

SECTIONS = [
    "1. Dataset yuklash",
    "2. Statistik tahlil (EDA)",
    "3. Data Cleaning",
    "4. Missing Values",
    "5. Outlier tahlili",
    "6. Feature Scaling",
    "7. Encoding usullari",
    "8. Model o'qitish",
    "9. Modelni baholash",
    "10. Encoding taqqoslash",
    "11. Yuklab olish",
]
section = st.sidebar.radio("Bosqichni tanlang:", SECTIONS, label_visibility="collapsed")

# Status indicator
st.sidebar.markdown("---")
if st.session_state.df_original is not None:
    st.sidebar.success(f"✓ Dataset: {st.session_state.df_original.shape[0]:,} × {st.session_state.df_original.shape[1]}")
    if st.session_state.target_col:
        st.sidebar.info(f"Target: **{st.session_state.target_col}**")
else:
    st.sidebar.warning("Dataset yuklanmagan")

st.sidebar.markdown("---")
st.sidebar.caption("Mavzu: Preprocessing va encoding usullarining AI aniqligiga ta'siri")


# =====================================================================
# SARLAVHA
# =====================================================================
st.markdown(f"""
<div style="border-bottom:3px solid {ACCENT};padding-bottom:10px;margin-bottom:20px">
  <h1 style="color:{PRIMARY};margin:0">Preprocessing va Encoding AI Model Aniqligiga Ta'siri</h1>
  <p style="color:{MUTED};margin:5px 0 0 0">Amaliy AI loyihasi · Telco Customer Churn dataseti misolida</p>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# 1. DATASET YUKLASH
# =====================================================================
if section.startswith("1."):
    st.header("1. Dataset yuklash")
    info("Bu bo'limda CSV faylni yuklang. Ustun turlari avtomatik aniqlanadi va target ustunni tanlang.")

    uploaded = st.file_uploader("CSV faylni yuklang", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df_original = df.copy()
            st.session_state.df_current = df.copy()
            num_cols, cat_cols = detect_column_types(df)
            st.session_state.numeric_cols = num_cols
            st.session_state.categorical_cols = cat_cols
            success_box(f"✓ Dataset muvaffaqiyatli yuklandi: <b>{df.shape[0]:,}</b> qator, <b>{df.shape[1]}</b> ustun")
        except Exception as e:
            st.error(f"Xato: faylni o'qib bo'lmadi. {e}")

    if st.session_state.df_original is not None:
        df = st.session_state.df_original

        st.subheader("Dataset o'lchami")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Qatorlar", f"{df.shape[0]:,}")
        c2.metric("Ustunlar", df.shape[1])
        c3.metric("Sonli ustunlar", len(st.session_state.numeric_cols))
        c4.metric("Kategoriyali", len(st.session_state.categorical_cols))

        st.subheader("Birinchi 5 qator")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Target ustunni tanlang")
        target_options = ["(tanlanmagan)"] + df.columns.tolist()
        current_idx = 0
        if st.session_state.target_col and st.session_state.target_col in df.columns:
            current_idx = target_options.index(st.session_state.target_col)
        target = st.selectbox("Target ustun:", target_options, index=current_idx)

        if target != "(tanlanmagan)":
            st.session_state.target_col = target
            # Targetni feature ro'yxatidan olib tashlaymiz
            num_cols, cat_cols = detect_column_types(df)
            if target in num_cols:
                num_cols.remove(target)
            if target in cat_cols:
                cat_cols.remove(target)
            st.session_state.numeric_cols = num_cols
            st.session_state.categorical_cols = cat_cols

            unique_targets = df[target].nunique()
            if unique_targets == 2:
                success_box(f"✓ Target: <b>{target}</b> · Binary classification ({unique_targets} sinf)")
            else:
                warn_box(f"Target: <b>{target}</b> · {unique_targets} sinf — multi-class")
        else:
            st.session_state.target_col = None
            warn_box("Target ustun tanlanmaguncha keyingi bosqichlar ishlamaydi.")


# =====================================================================
# 2. EDA
# =====================================================================
elif section.startswith("2."):
    st.header("2. Dastlabki statistik tahlil (EDA)")

    if st.session_state.df_current is None:
        warn_box("Avval dataset yuklang (1-bo'lim).")
    else:
        df = st.session_state.df_current
        target = st.session_state.target_col

        # Info table
        st.subheader("Ustunlar tahlili (df.info)")
        info_data = []
        for col in df.columns:
            info_data.append({
                "Ustun": col,
                "Turi": str(df[col].dtype),
                "Bo'sh qiymatlar": df[col].isna().sum(),
                "Bo'sh foiz (%)": round(df[col].isna().sum() / len(df) * 100, 2),
                "Noyob qiymatlar": df[col].nunique(),
            })
        st.dataframe(pd.DataFrame(info_data), use_container_width=True)

        # Describe
        st.subheader("Sonli ustunlar statistikasi (df.describe)")
        if st.session_state.numeric_cols:
            st.dataframe(df[st.session_state.numeric_cols].describe().round(3), use_container_width=True)
        else:
            info("Sonli ustun topilmadi.")

        # Missing summary
        st.subheader("Missing values xulosasi")
        total_missing = df.isna().sum().sum()
        c1, c2 = st.columns(2)
        c1.metric("Jami bo'sh qiymatlar", f"{total_missing:,}")
        c2.metric("Bo'sh qiymatli ustunlar", (df.isna().sum() > 0).sum())

        # Duplicates
        st.subheader("Takrorlanuvchi qatorlar")
        dup_count = df.duplicated().sum()
        if dup_count == 0:
            success_box(f"✓ Takror qator yo'q.")
        else:
            warn_box(f"⚠ <b>{dup_count}</b> ta takror qator topildi. \"Data Cleaning\" bo'limida olib tashlash mumkin.")

        # Target distribution
        if target:
            st.subheader(f"Target taqsimoti — `{target}`")
            target_counts = df[target].value_counts()
            target_pct = (target_counts / len(df) * 100).round(2)

            fig = px.bar(
                x=target_counts.index.astype(str), y=target_counts.values,
                labels={"x": target, "y": "Qatorlar soni"},
                color=target_counts.index.astype(str),
                color_discrete_sequence=[PRIMARY, ACCENT, SUCCESS, DANGER, MUTED],
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Disbalans tekshirish (binary uchun)
            if df[target].nunique() == 2:
                max_pct = target_pct.max()
                min_pct = target_pct.min()
                if max_pct - min_pct > 30:
                    warn_box(
                        f"⚠ <b>Target disbalansli</b> — sinflar foizi: " +
                        " | ".join([f"{v}: {p}%" for v, p in target_pct.items()]) +
                        ". Accuracy yetarli emas — <b>Recall va F1-score</b> muhim."
                    )
                else:
                    info(f"Sinflar nisbatan balanced: " + " | ".join([f"{v}: {p}%" for v, p in target_pct.items()]))

        # Sonli ustunlar histogrami
        if st.session_state.numeric_cols:
            st.subheader("Sonli ustunlar tarqalishi (histogram)")
            sel = st.multiselect("Ustunlarni tanlang:", st.session_state.numeric_cols,
                                 default=st.session_state.numeric_cols[:min(3, len(st.session_state.numeric_cols))])
            for col in sel:
                fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=[PRIMARY])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        # Kategoriyali ustunlar value counts
        if st.session_state.categorical_cols:
            st.subheader("Kategoriyali ustunlar — value counts")
            cat_sel = st.selectbox("Ustunni tanlang:", st.session_state.categorical_cols)
            if cat_sel:
                vc = df[cat_sel].value_counts()
                fig = px.bar(x=vc.index.astype(str), y=vc.values,
                             labels={"x": cat_sel, "y": "Soni"},
                             color_discrete_sequence=[SECONDARY := "#2E75B6"])
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# 3. DATA CLEANING
# =====================================================================
elif section.startswith("3."):
    st.header("3. Data Cleaning")
    info("Takror qatorlar, ortiqcha bo'sh joylar va yashirin bo'sh qiymatlar tozalanadi.")

    if st.session_state.df_current is None:
        warn_box("Avval dataset yuklang.")
    else:
        df = st.session_state.df_current.copy()

        # Duplicates
        st.subheader("Takror qatorlar")
        dup_count = df.duplicated().sum()
        c1, c2 = st.columns([2, 1])
        c1.metric("Takror qatorlar soni", dup_count)
        if c2.button("Takrorlarni olib tashlash", disabled=(dup_count == 0)):
            df = df.drop_duplicates().reset_index(drop=True)
            st.session_state.df_current = df
            success_box(f"✓ {dup_count} ta takror qator olib tashlandi.")
            st.rerun()

        # Whitespace + hidden NaN
        st.subheader("Object ustunlardagi bo'sh joylarni tozalash")
        info("Bo'sh joylarni olib tashlash (strip) va yashirin bo'sh qiymatlarni NaN ga aylantirish: \"\", \" \", \"?\", \"NA\", \"N/A\", \"null\".")
        if st.button("Tozalashni boshlash"):
            cleaned = 0
            hidden_nan_values = {"", " ", "?", "NA", "N/A", "null", "None", "nan"}
            for col in df.select_dtypes(include=["object"]).columns:
                before_na = df[col].isna().sum()
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(list(hidden_nan_values), np.nan)
                after_na = df[col].isna().sum()
                cleaned += (after_na - before_na)
            st.session_state.df_current = df
            success_box(f"✓ Tozalash bajarildi. Qo'shimcha {cleaned} ta yashirin NaN topildi.")
            st.rerun()

        # Numeric conversion
        st.subheader("Ustunni numeric formatga o'tkazish")
        info("Telco Customer Churn datasetida TotalCharges ustuni object turida saqlangan. Quyidagi tugma uni avtomatik aniqlab, raqamli formatga o'tkazadi.")

        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if "TotalCharges" in obj_cols:
            if st.button("TotalCharges ni numeric ga aylantirish"):
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                st.session_state.df_current = df
                num_cols, cat_cols = detect_column_types(df)
                if st.session_state.target_col in num_cols:
                    num_cols.remove(st.session_state.target_col)
                if st.session_state.target_col in cat_cols:
                    cat_cols.remove(st.session_state.target_col)
                st.session_state.numeric_cols = num_cols
                st.session_state.categorical_cols = cat_cols
                nan_count = df["TotalCharges"].isna().sum()
                success_box(f"✓ TotalCharges raqamli. Yangi NaN: {nan_count} ta (Missing Values bo'limida to'ldiring).")
                st.rerun()

        manual_col = st.selectbox("Yoki boshqa ustunni tanlang:", ["(tanlanmagan)"] + obj_cols)
        if manual_col != "(tanlanmagan)" and st.button(f"`{manual_col}` ni numeric ga aylantirish"):
            df[manual_col] = pd.to_numeric(df[manual_col], errors="coerce")
            st.session_state.df_current = df
            num_cols, cat_cols = detect_column_types(df)
            if st.session_state.target_col in num_cols:
                num_cols.remove(st.session_state.target_col)
            if st.session_state.target_col in cat_cols:
                cat_cols.remove(st.session_state.target_col)
            st.session_state.numeric_cols = num_cols
            st.session_state.categorical_cols = cat_cols
            success_box(f"✓ `{manual_col}` raqamli formatga o'tkazildi.")
            st.rerun()

        # Joriy holat
        st.subheader("Joriy dataset (cleaning'dan keyin)")
        st.dataframe(df.head(), use_container_width=True)


# =====================================================================
# 4. MISSING VALUES
# =====================================================================
elif section.startswith("4."):
    st.header("4. Missing Values bilan ishlash")

    if st.session_state.df_current is None:
        warn_box("Avval dataset yuklang.")
    else:
        df = st.session_state.df_current.copy()

        # Missing jadval
        st.subheader("Bo'sh qiymatlar bo'yicha ustunlar")
        miss_data = []
        for col in df.columns:
            cnt = df[col].isna().sum()
            if cnt > 0:
                miss_data.append({"Ustun": col, "NaN soni": cnt, "Foiz (%)": round(cnt / len(df) * 100, 2)})
        if not miss_data:
            success_box("✓ Bo'sh qiymat yo'q.")
        else:
            miss_df = pd.DataFrame(miss_data)
            st.dataframe(miss_df, use_container_width=True)

            # Foiz ko'rsatkichlariga qarab izoh
            max_pct = miss_df["Foiz (%)"].max()
            if max_pct < 5:
                info(f"Eng ko'p NaN: <b>{max_pct}%</b>. Drop qilish ham mumkin, lekin ma'lumotni saqlash uchun to'ldirish ham yaxshi tanlov.")
            else:
                warn_box(f"⚠ Eng ko'p NaN: <b>{max_pct}%</b>. Darrov drop qilish tavsiya etilmaydi — ko'p ma'lumot yo'qoladi.")

            st.subheader("To'ldirish usulini tanlang")
            method = st.radio("Yondashuv:", [
                "Drop rows — bo'sh qiymat bor qatorlarni o'chirish",
                "Mean bilan to'ldirish — faqat sonli ustunlar",
                "Median bilan to'ldirish — faqat sonli ustunlar",
                "Mode bilan to'ldirish — kategoriyali ustunlar",
                "Avtomatik — sonli (median) + kategoriyali (mode)",
            ])

            if st.button("Qo'llash"):
                if method.startswith("Drop"):
                    before = len(df)
                    df = df.dropna().reset_index(drop=True)
                    success_box(f"✓ {before - len(df)} ta qator o'chirildi.")
                elif method.startswith("Mean"):
                    for col in df.select_dtypes(include=[np.number]).columns:
                        df[col] = df[col].fillna(df[col].mean())
                    success_box("✓ Sonli ustunlar mean bilan to'ldirildi.")
                elif method.startswith("Median"):
                    for col in df.select_dtypes(include=[np.number]).columns:
                        df[col] = df[col].fillna(df[col].median())
                    success_box("✓ Sonli ustunlar median bilan to'ldirildi.")
                elif method.startswith("Mode"):
                    for col in df.select_dtypes(include=["object", "category"]).columns:
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val[0])
                    success_box("✓ Kategoriyali ustunlar mode bilan to'ldirildi.")
                else:  # Avtomatik
                    for col in df.select_dtypes(include=[np.number]).columns:
                        df[col] = df[col].fillna(df[col].median())
                    for col in df.select_dtypes(include=["object", "category"]).columns:
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val[0])
                    success_box("✓ Avtomatik: sonli → median, kategoriyali → mode.")
                st.session_state.df_current = df
                st.rerun()


# =====================================================================
# 5. OUTLIER TAHLILI
# =====================================================================
elif section.startswith("5."):
    st.header("5. Outlier tahlili (IQR usuli)")
    info("IQR = Q3 - Q1. Outlier: <code>x &lt; Q1 - 1.5·IQR</code> yoki <code>x &gt; Q3 + 1.5·IQR</code>.")
    info("<b>Random Forest, XGBoost</b> kabi tree modellari outlierga kuchli sezgir emas — alohida ishlov shart emas.")

    if st.session_state.df_current is None:
        warn_box("Avval dataset yuklang.")
    else:
        df = st.session_state.df_current.copy()
        numeric_cols = [c for c in st.session_state.numeric_cols if c in df.columns]

        if not numeric_cols:
            info("Sonli ustun topilmadi.")
        else:
            outlier_info = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask = (df[col] < lower) | (df[col] > upper)
                outlier_info[col] = {
                    "Q1": Q1, "Q3": Q3, "IQR": IQR,
                    "lower": lower, "upper": upper,
                    "outlier_count": mask.sum(),
                    "outlier_pct": round(mask.sum() / len(df) * 100, 2),
                }

            st.subheader("Outlier xulosasi")
            summary_df = pd.DataFrame([
                {"Ustun": c, "Q1": round(d["Q1"], 2), "Q3": round(d["Q3"], 2),
                 "IQR": round(d["IQR"], 2), "Outlier soni": d["outlier_count"],
                 "Outlier %": d["outlier_pct"]}
                for c, d in outlier_info.items()
            ])
            st.dataframe(summary_df, use_container_width=True)

            st.subheader("Boxplot")
            sel = st.selectbox("Ustun:", numeric_cols)
            fig = px.box(df, y=sel, color_discrete_sequence=[PRIMARY])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Outlier bilan nima qilish?")
            action = st.radio("Tanlov:", [
                "Qoldirish (Random Forest da odatda zarari yo'q)",
                "Olib tashlash (drop)",
                "Cap qilish (chegara qiymatiga almashtirish)",
            ])

            apply_to = st.multiselect("Qaysi ustunlarga qo'llash:", numeric_cols, default=numeric_cols)
            if st.button("Qo'llash"):
                if action.startswith("Qoldirish"):
                    info("Hech qanday o'zgartirish kiritilmadi.")
                elif action.startswith("Olib tashlash"):
                    before = len(df)
                    mask = pd.Series(True, index=df.index)
                    for col in apply_to:
                        d = outlier_info[col]
                        mask &= (df[col] >= d["lower"]) & (df[col] <= d["upper"])
                    df = df[mask].reset_index(drop=True)
                    st.session_state.df_current = df
                    success_box(f"✓ {before - len(df)} ta outlier qator olib tashlandi.")
                    st.rerun()
                else:  # Cap
                    capped = 0
                    for col in apply_to:
                        d = outlier_info[col]
                        before_cnt = ((df[col] < d["lower"]) | (df[col] > d["upper"])).sum()
                        df[col] = df[col].clip(lower=d["lower"], upper=d["upper"])
                        capped += before_cnt
                    st.session_state.df_current = df
                    success_box(f"✓ {capped} ta qiymat chegara qiymatiga cap qilindi.")
                    st.rerun()


# =====================================================================
# 6. FEATURE SCALING
# =====================================================================
elif section.startswith("6."):
    st.header("6. Feature Scaling")
    info("Sonli ustunlarni bir xil shkalaga keltirish. Scaling faqat sonli ustunlarga qo'llanadi.")
    info("<b>Linear/Logistic Regression, KNN, SVM, NN</b> uchun scaling muhim. <b>Random Forest, XGBoost</b> uchun majburiy emas.")

    if st.session_state.df_current is None:
        warn_box("Avval dataset yuklang.")
    else:
        df = st.session_state.df_current.copy()
        numeric_cols = [c for c in st.session_state.numeric_cols if c in df.columns]

        method = st.radio("Scaler tanlang:", ["Scaling qilmaslik", "StandardScaler", "MinMaxScaler"])

        if numeric_cols and method != "Scaling qilmaslik":
            if st.button("Qo'llash"):
                if method == "StandardScaler":
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.session_state.df_current = df
                success_box(f"✓ {method} qo'llandi ({len(numeric_cols)} ta sonli ustunga).")
                st.rerun()

        st.subheader("Sonli ustunlar holati")
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)


# =====================================================================
# 7. ENCODING
# =====================================================================
elif section.startswith("7."):
    st.header("7. Encoding usullari")
    info("Bu — loyihaning markaziy bo'limi. Kategoriyali ustunlarni raqamga aylantirish. 4 ta usul mavjud.")

    with st.expander("📖 Har bir usul haqida qisqacha"):
        st.markdown(f"""
        **Label Encoding** — Kategoriyalarga 0, 1, 2 kabi sonlar beriladi. Binary yoki ordinal ustunlar uchun qulay, lekin nominal ustunlarda soxta tartib hosil qilishi mumkin.

        **One-Hot Encoding** — Har bir kategoriya alohida 0/1 ustunga aylantiriladi. Nominal ustunlar uchun yaxshi, lekin kategoriya ko'p bo'lsa ustunlar soni oshib ketadi.

        **Frequency Encoding** — Kategoriya datasetda necha marta uchrashiga qarab raqamga almashtiriladi. High cardinality ustunlarda foydali.

        **Target Encoding** — Kategoriya shu kategoriyadagi target o'rtacha qiymati bilan almashtiriladi. Kuchli usul, lekin <span style="color:{DANGER}"><b>data leakage</b></span> xavfi bor. Bu ilovada K-Fold target encoding ishlatiladi (leakage'siz).
        """, unsafe_allow_html=True)

    if st.session_state.df_current is None or st.session_state.target_col is None:
        warn_box("Avval dataset yuklang va target ustunni tanlang.")
    else:
        df = st.session_state.df_current
        target = st.session_state.target_col
        cat_cols_all = [c for c in st.session_state.categorical_cols if c in df.columns and c != target]
        num_cols = [c for c in st.session_state.numeric_cols if c in df.columns and c != target]

        # customerID kabi yuqori kardinallik (ID) ustunlarni olib tashlaymiz
        id_threshold = max(10, int(len(df) * 0.5))
        id_cols = [c for c in cat_cols_all if df[c].nunique() >= id_threshold]
        cat_cols = [c for c in cat_cols_all if c not in id_cols]
        if id_cols:
            st.info(f"⚠️ ID ustunlar (encoding uchun o'tkazildi): {', '.join(id_cols)}")

        st.subheader("Encoding usulini tanlang")
        method = st.selectbox("Usul:", ["Label Encoding", "One-Hot Encoding", "Frequency Encoding", "Target Encoding"])

        st.markdown(f"**Kategoriyali ustunlar ({len(cat_cols)}):** {', '.join(cat_cols) if cat_cols else 'topilmadi'}")

        if not cat_cols:
            warn_box("Kategoriyali ustun yo'q — encoding qilishga hojat yo'q.")
        else:
            if st.button("Encoding qo'llash"):
                X = df[num_cols + cat_cols].copy()
                y = df[target].copy()
                # Pandas 2.x: StringDtype, object, category hammasini numeric ga o'giramiz
                if not pd.api.types.is_numeric_dtype(y):
                    y, _ = pd.factorize(y)
                    y = pd.Series(y)

                # Faqat encoding namunasini ko'rsatish uchun — train/val/test split + encoding
                strat = y if len(np.unique(y)) > 1 else None
                X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=strat)
                strat_tv = y_tv if len(np.unique(y_tv)) > 1 else None
                X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176, random_state=42, stratify=strat_tv)

                if method == "Label Encoding":
                    Xtr_e, Xv_e, Xte_e = label_encode_splits(X_train, X_val, X_test, cat_cols)
                elif method == "One-Hot Encoding":
                    Xtr_e, Xv_e, Xte_e = onehot_encode_splits(X_train, X_val, X_test, cat_cols)
                elif method == "Frequency Encoding":
                    Xtr_e, Xv_e, Xte_e = frequency_encode_splits(X_train, X_val, X_test, cat_cols)
                else:
                    Xtr_e, Xv_e, Xte_e = target_encode_splits(X_train, pd.Series(y_train), X_val, X_test, cat_cols)

                st.session_state.encoded_df = Xtr_e.copy()
                st.session_state["_encoding_method"] = method
                st.session_state["_split_data"] = {
                    "X_train": Xtr_e, "X_val": Xv_e, "X_test": Xte_e,
                    "y_train": y_train, "y_val": y_val, "y_test": y_test,
                }
                success_box(f"✓ <b>{method}</b> qo'llandi. Train: {len(Xtr_e)} qator, ustunlar: {Xtr_e.shape[1]}")

            if st.session_state.encoded_df is not None:
                st.subheader("Encoding natijasi (train datasetining birinchi 5 qatori)")
                st.dataframe(st.session_state.encoded_df.head(), use_container_width=True)


# =====================================================================
# 8. MODEL O'QITISH
# =====================================================================
elif section.startswith("8."):
    st.header("8. Model o'qitish")

    if st.session_state.df_current is None or st.session_state.target_col is None:
        warn_box("Avval dataset yuklang va target ustunni tanlang.")
    elif st.session_state.encoded_df is None:
        warn_box("Avval encoding qo'llang (7-bo'lim).")
    else:
        df = st.session_state.df_current
        target = st.session_state.target_col

        st.subheader("Train / Validation / Test ajratish")
        info("Tartib: <b>70% train · 15% validation · 15% test</b> · stratified · random_state=42")

        split_data = st.session_state["_split_data"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Train", f"{len(split_data['X_train']):,}")
        c2.metric("Validation", f"{len(split_data['X_val']):,}")
        c3.metric("Test", f"{len(split_data['X_test']):,}")

        st.subheader("Model tanlash")
        model_name = st.selectbox("Model:", ["Random Forest", "Logistic Regression",
                                              "Decision Tree", "KNN", "Naive Bayes"], index=0)
        if model_name == "Random Forest":
            info("✓ <b>Random Forest</b> ko'plab Decision Tree lardan iborat ensemble model bo'lib, yuqori aniqlik, barqarorlik va overfittingga chidamliligi sababli tanlandi. Shuningdek, <b>feature importance</b> orqali qaysi ustunlar muhimligini ko'rsatadi.")

        # Hyperparameter tuning
        st.subheader("Giperparametrlarni sozlash (Random Forest)")
        use_grid = st.checkbox("GridSearchCV bilan tuning bajarish", value=False,
                                disabled=(model_name != "Random Forest"))
        if use_grid and model_name == "Random Forest":
            info("Sozlanadi: n_estimators [100, 200, 300] · max_depth [5, 10, 15, None] · min_samples_split [2, 5, 10] · max_features [sqrt, log2]. Scoring: F1 · CV: 5-fold.")

        # O'qitish tugmasi
        if st.button("Modelni o'qitish"):
            X_train, X_val, X_test = split_data["X_train"], split_data["X_val"], split_data["X_test"]
            y_train, y_val, y_test = split_data["y_train"], split_data["y_val"], split_data["y_test"]

            with st.spinner("Model o'qitilmoqda..."):
                if use_grid and model_name == "Random Forest":
                    param_grid = {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [5, 10, 15, None],
                        "min_samples_split": [2, 5, 10],
                        "max_features": ["sqrt", "log2"],
                    }
                    gs = GridSearchCV(
                        RandomForestClassifier(random_state=42, n_jobs=-1),
                        param_grid, scoring="f1_weighted", cv=5, n_jobs=-1, verbose=0,
                    )
                    gs.fit(X_train, y_train)
                    model = gs.best_estimator_
                    st.session_state.best_params = gs.best_params_
                    success_box(f"✓ <b>Eng yaxshi parametrlar:</b> {gs.best_params_}<br><b>CV F1-score:</b> {gs.best_score_:.4f}")
                else:
                    model = get_classifier(model_name)
                    model.fit(X_train, y_train)

                st.session_state.trained_model = model

                # Feature importance (RF)
                if hasattr(model, "feature_importances_"):
                    fi_df = pd.DataFrame({
                        "Feature": X_train.columns,
                        "Importance": model.feature_importances_,
                    }).sort_values("Importance", ascending=False)
                    st.session_state.feature_importance_df = fi_df

            success_box(f"✓ <b>{model_name}</b> o'qitildi. Endi 9-bo'limda baholang.")


# =====================================================================
# 9. MODELNI BAHOLASH
# =====================================================================
elif section.startswith("9."):
    st.header("9. Modelni baholash (Test set)")

    if st.session_state.trained_model is None:
        warn_box("Avval modelni o'qiting (8-bo'lim).")
    else:
        model = st.session_state.trained_model
        split_data = st.session_state["_split_data"]
        X_test, y_test = split_data["X_test"], split_data["y_test"]

        # Metrics
        metrics = evaluate_classifier(model, X_test, y_test)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        c2.metric("Precision", f"{metrics['precision']:.4f}")
        c3.metric("Recall", f"{metrics['recall']:.4f}")
        c4.metric("F1-score", f"{metrics['f1']:.4f}")

        # Disbalans izoh
        if pd.Series(y_test).nunique() == 2:
            counts = pd.Series(y_test).value_counts()
            if counts.max() / counts.sum() > 0.65:
                warn_box("⚠ Disbalansli dataset — <b>Accuracy aldamchi bo'lishi mumkin</b>. Ketadigan mijozlarni topish uchun <b>Recall va F1-score</b> muhim.")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, metrics["y_pred"])
        unique_labels = sorted(pd.Series(y_test).unique())
        cm_df = pd.DataFrame(cm, index=[f"Haqiqiy: {l}" for l in unique_labels],
                              columns=[f"Bashorat: {l}" for l in unique_labels])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Bashorat", y="Haqiqiy", color="Soni"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Classification report
        st.subheader("Classification Report")
        rep = classification_report(y_test, metrics["y_pred"], output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(rep).T.round(4)
        st.dataframe(rep_df, use_container_width=True)

        # Feature importance
        if st.session_state.feature_importance_df is not None:
            st.subheader("Feature Importance")
            info("Feature importance model qaroriga qaysi ustunlar kuchli ta'sir qilganini ko'rsatadi.")
            fi_df = st.session_state.feature_importance_df.head(10)
            fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=[ACCENT])
            fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(st.session_state.feature_importance_df, use_container_width=True)


# =====================================================================
# 10. ENCODING TAQQOSLASH
# =====================================================================
elif section.startswith("10."):
    st.header("10. Encoding usullarini taqqoslash")
    info("4 ta encoding usulini bir xil splitda, bir xil modelda (Random Forest) sinab, natijalarni taqqoslaymiz. Bu — loyihaning asosiy qismi hisoblanadi")

    if st.session_state.df_current is None or st.session_state.target_col is None:
        warn_box("Avval dataset yuklang va target ustunni tanlang.")
    else:
        df = st.session_state.df_current
        target = st.session_state.target_col
        cat_cols = [c for c in st.session_state.categorical_cols if c in df.columns and c != target]
        num_cols = [c for c in st.session_state.numeric_cols if c in df.columns and c != target]

        if not cat_cols:
            warn_box("Kategoriyali ustun yo'q — encoding taqqoslash uchun ma'no yo'q.")
        else:
            model_choice = st.selectbox("Taqqoslash modeli:", ["Random Forest", "Logistic Regression", "Decision Tree", "KNN"], index=0)
            apply_scaling = st.checkbox("Linear/KNN uchun scaling qo'llash (StandardScaler)",
                                         value=(model_choice in {"Logistic Regression", "KNN"}))

            if st.button("Taqqoslashni boshlash"):
                results = []
                with st.spinner("4 ta usul ketma-ket sinovdan o'tkazilmoqda..."):
                    for enc_method in ["Label Encoding", "One-Hot Encoding", "Frequency Encoding", "Target Encoding"]:
                        try:
                            res = run_encoding_pipeline(
                                df, target, cat_cols, num_cols, enc_method,
                                model_name=model_choice, apply_scaling=apply_scaling,
                            )
                            results.append({
                                "Encoding": enc_method,
                                "Accuracy": round(res["test_metrics"]["accuracy"], 4),
                                "Precision": round(res["test_metrics"]["precision"], 4),
                                "Recall": round(res["test_metrics"]["recall"], 4),
                                "F1-score": round(res["test_metrics"]["f1"], 4),
                            })
                        except Exception as e:
                            results.append({
                                "Encoding": enc_method, "Accuracy": None, "Precision": None,
                                "Recall": None, "F1-score": None, "_error": str(e),
                            })

                comp_df = pd.DataFrame(results)
                st.session_state.comparison_df = comp_df

            if st.session_state.comparison_df is not None:
                comp_df = st.session_state.comparison_df
                st.subheader("Taqqoslash natijalari")
                st.dataframe(comp_df, use_container_width=True)

                # F1 bar chart
                clean_df = comp_df.dropna(subset=["F1-score"]).copy()
                if not clean_df.empty:
                    fig = px.bar(clean_df, x="Encoding", y="F1-score",
                                 color="Encoding",
                                 color_discrete_sequence=[PRIMARY, "#2E75B6", ACCENT, SUCCESS],
                                 text="F1-score")
                    fig.update_traces(textposition="outside")
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # Eng yaxshi
                    best_row = clean_df.loc[clean_df["F1-score"].idxmax()]
                    success_box(f"✓ <b>Eng yaxshi natija:</b> {best_row['Encoding']} · <b>F1-score = {best_row['F1-score']:.4f}</b>")

                    if best_row["Encoding"] == "Target Encoding":
                        warn_box("⚠ Target Encoding kuchli natija berishi mumkin, lekin <b>leakage xavfi</b> sababli uni doim train yoki cross-validation ichida hisoblash kerak. Bu ilovada K-Fold target encoding ishlatildi — leakage'siz.")


# =====================================================================
# 11. YUKLAB OLISH
# =====================================================================
elif section.startswith("11."):
    st.header("11. Natijalarni yuklab olish")

    if st.session_state.df_current is None:
        warn_box("Avval dataset yuklang.")
    else:
        df = st.session_state.df_current

        # CSV - preprocessed
        st.subheader("Preprocessing qilingan dataset")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 preprocessed_dataset.csv", csv_bytes, "preprocessed_dataset.csv", "text/csv")

        # CSV - encoded
        if st.session_state.encoded_df is not None:
            st.subheader("Encoding qilingan dataset")
            enc_bytes = st.session_state.encoded_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 encoded_dataset.csv", enc_bytes, "encoded_dataset.csv", "text/csv")

        # Comparison results
        if st.session_state.comparison_df is not None:
            st.subheader("Encoding taqqoslash natijalari")
            comp_bytes = st.session_state.comparison_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 encoding_comparison.csv", comp_bytes, "encoding_comparison.csv", "text/csv")

        # Trained model
        if st.session_state.trained_model is not None:
            st.subheader("O'qitilgan model")
            buf = io.BytesIO()
            joblib.dump(st.session_state.trained_model, buf)
            buf.seek(0)
            st.download_button("📥 trained_model.joblib", buf.getvalue(), "trained_model.joblib", "application/octet-stream")

        # Report
        st.subheader("Qisqa hisobot")
        lines = [
            "PREPROCESSING VA ENCODING — LOYIHA HISOBOTI",
            "=" * 60,
            f"Muallif: Tursunov Sherzod Abduvakil ugli",
            f"Sana: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "DATASET",
            f"  Qatorlar: {len(df):,}",
            f"  Ustunlar: {df.shape[1]}",
            f"  Target: {st.session_state.target_col or '(tanlanmagan)'}",
            f"  Sonli ustunlar: {len(st.session_state.numeric_cols)}",
            f"  Kategoriyali ustunlar: {len(st.session_state.categorical_cols)}",
            "",
        ]
        if st.session_state.best_params:
            lines.append("GRIDSEARCHCV NATIJASI")
            lines.append(f"  best_params: {st.session_state.best_params}")
            lines.append("")
        if st.session_state.comparison_df is not None:
            lines.append("ENCODING TAQQOSLASH (F1-score)")
            for _, row in st.session_state.comparison_df.iterrows():
                lines.append(f"  {row['Encoding']:25s}  F1 = {row['F1-score']}")
        report = "\n".join(lines)
        st.text_area("Hisobot ko'rinishi:", report, height=300)
        st.download_button("📥 report.txt", report.encode("utf-8"), "report.txt", "text/plain")
