import html
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Analisis Sentimen Ulasan",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Aplikasi Analisis Sentimen Ulasan")
st.markdown(
    "Aplikasi satu halaman untuk visualisasi data, performa model, dan prediksi interaktif."
)
st.markdown("---")

# --------------------------------------------------------------------------
# Konstanta & konfigurasi pra-pemrosesan
# --------------------------------------------------------------------------

# Stopwords yang benar-benar aman (KATA NEGASI TIDAK DIHAPUS agar makna kalimat
# tidak berubah, mis. "tidak sesuai" tidak boleh menjadi "sesuai")
INDONESIAN_STOPWORDS = {
    "dan", "saya", "yg", "di", "nya", "yang", "ini", "itu", "ada",
    "untuk", "with", "dengan", "ke", "dari", "ya", "aja", "pake",
    "sudah", "udah", "juga", "buat", "gan", "si",
    "sama", "cuma", "sekali", "udahsudah", "terima", "kasih", "terimakasih",
    "ok", "pas", "d", "apa", "kamu", "kami", "dg", "the",
    "atau", "pada", "adalah", "akan", "karena",
}

# Normalisasi singkatan negasi -> bentuk baku "tidak" (kata negasi TIDAK
# dihapus sebagai stopword karena menentukan polaritas kalimat).
NEGATION_MAP = {
    "nggak": "tidak", "ngga": "tidak", "kagak": "tidak",
    "gak": "tidak", "gx": "tidak", "tdk": "tidak", "tak": "tidak",
    "ga": "tidak",
}

# Normalisasi singkatan umum lain -> bentuk baku. Ini penting agar mis.
# "barang" dan "brg" dihitung sebagai token TF-IDF yang SAMA, bukan dua fitur
# terpisah yang masing-masing jarang muncul (memperlemah sinyal model).
SLANG_MAP = {
    "dgn": "dengan", "yg": "yang", "tpi": "tapi", "tp": "tapi",
    "sdh": "sudah", "udh": "sudah", "brg": "barang", "bgt": "banget",
    "krn": "karena", "blm": "belum", "jgn": "jangan", "jd": "jadi",
    "dr": "dari", "trs": "terus", "utk": "untuk", "sy": "saya",
    "skrg": "sekarang", "dpt": "dapat", "gmn": "gimana", "bgmn": "gimana",
    "org": "orang", "cust": "customer", "sblm": "sebelum", "stlh": "setelah",
}

CATEGORY_LABELS = {0: "Negatif", 1: "Positif"}


def bersihkan_teks(text: str) -> str:
    """Membersihkan satu baris teks ulasan.

    Tahapan: decode HTML entity (mis. ``&#34;``) -> lowercase -> normalisasi
    kata negasi -> hapus karakter non-huruf -> pangkas huruf berulang
    berlebihan (mis. "baguuus" -> "baguus") -> hapus stopword & token
    sepanjang 1 huruf.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = text.lower()

    for singkatan, baku in NEGATION_MAP.items():
        text = re.sub(rf"\b{singkatan}\b", baku, text)

    for singkatan, baku in SLANG_MAP.items():
        text = re.sub(rf"\b{singkatan}\b", baku, text)

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # "sangaaaat" -> "sangaat"

    words = [
        w for w in text.split()
        if w not in INDONESIAN_STOPWORDS and len(w) > 1
    ]
    return " ".join(words)


# --------------------------------------------------------------------------
# Loading & caching data / model
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Membaca CSV dari bytes (aman untuk cache).

    Mencoba beberapa encoding secara berurutan karena banyak file CSV hasil
    ekspor Excel di Indonesia tidak disimpan dalam UTF-8, melainkan
    Windows-1252 / Latin-1, yang bila dipaksa dibaca sebagai UTF-8 akan
    menghasilkan UnicodeDecodeError.
    """
    import io

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    raise last_error


@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Ulasan", "label"]).copy()
    df["Ulasan"] = df["Ulasan"].astype(str)
    df["Ulasan_Clean"] = df["Ulasan"].apply(bersihkan_teks)
    df = df[df["Ulasan_Clean"].str.strip() != ""].copy()
    return df


@st.cache_resource(show_spinner=False)
def train_model(X_train, y_train):
    """Melatih pipeline TF-IDF + Multinomial Naive Bayes.

    Parameter TF-IDF di-tuning (min_df, sublinear_tf) untuk mengurangi noise
    dari n-gram yang sangat jarang muncul dan menstabilkan pembobotan.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("naive_bayes", MultinomialNB()),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


@st.cache_data(show_spinner=False)
def cross_validate(_pipeline, X, y):
    """Skor cross-validation 5-fold agar evaluasi tidak bergantung pada satu
    kombinasi train/test split saja."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(_pipeline, X, y, cv=cv, scoring="accuracy")
    return scores


# --------------------------------------------------------------------------
# 1. Eksplorasi Data Awal
# --------------------------------------------------------------------------

st.header("1. Eksplorasi Data Awal")

uploaded_file = st.file_uploader(
    "Unggah dataset ulasan (.csv) — kosongkan untuk memakai data_real.csv bawaan",
    type=["csv"],
)

try:
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        st.caption(f"Menggunakan file unggahan: **{uploaded_file.name}**")
    else:
        with open("data_real.csv", "rb") as f:
            file_bytes = f.read()
        st.caption("Menggunakan dataset bawaan: **data_real.csv**")
except FileNotFoundError:
    st.error(
        "File 'data_real.csv' tidak ditemukan. Pastikan file berada di folder "
        "yang sama dengan skrip ini, atau unggah file CSV Anda sendiri di atas."
    )
    st.stop()

try:
    df_raw = load_data(file_bytes)
except pd.errors.EmptyDataError:
    st.error("File CSV kosong atau tidak dapat dibaca. Silakan unggah file lain.")
    st.stop()
except pd.errors.ParserError:
    st.error("File CSV tidak valid / gagal diparsing. Pastikan formatnya benar.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataframe Asli (20 Baris Pertama)")
    st.dataframe(df_raw.head(20), width="stretch")

with col2:
    keep_cols = ["Ulasan", "label"]
    df = df_raw[[c for c in df_raw.columns if c in keep_cols]].copy()
    st.subheader("Dataframe Setelah Filter Kolom")
    st.dataframe(df.head(20), width="stretch")

if "label" not in df.columns or "Ulasan" not in df.columns:
    st.error("Dataset harus memiliki kolom 'Ulasan' dan 'label'.")
    st.stop()

# --------------------------------------------------------------------------
# 2. Pembersihan & Pra-pemrosesan Data
# --------------------------------------------------------------------------

st.header("2. Pembersihan & Pra-pemrosesan Data")

with st.spinner("Sedang membersihkan data..."):
    df = preprocess_data(df)

if df.empty:
    st.warning("Tidak ada data tersisa setelah pembersihan. Periksa kembali dataset Anda.")
    st.stop()

st.success(f"Data Preprocessing Selesai! ({len(df)} baris siap dianalisis)")
st.dataframe(df[["Ulasan", "Ulasan_Clean", "label"]].head(10), width="stretch")

st.download_button(
    "⬇️ Unduh Data Bersih (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="data_bersih.csv",
    mime="text/csv",
)

# --------------------------------------------------------------------------
# 3. Visualisasi & Pembagian Data
# --------------------------------------------------------------------------

st.header("3. Visualisasi & Pembagian Data")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("WordCloud Kata Populer")
    all_text = " ".join(review for review in df["Ulasan_Clean"])

    if all_text.strip() != "":
        wordcloud = WordCloud(
            background_color="white",
            width=800,
            height=400,
            max_words=100,
            colormap="viridis",
        ).generate(all_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Tidak ada kata yang cukup untuk menghasilkan WordCloud.")

with col_right:
    st.subheader("Metrik Pembagian Data")

    X = df["Ulasan_Clean"]
    y = df["label"]

    if y.nunique() < 2:
        st.error("Kolom label harus memiliki minimal 2 kelas untuk klasifikasi.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    st.metric(label="Total Baris Dataset", value=f"{len(df)} baris")
    st.metric(label="Data Latih (80%)", value=f"{len(X_train)} baris")
    st.metric(label="Data Uji (20%)", value=f"{len(X_test)} baris")

    st.caption("Distribusi Label")
    st.bar_chart(df["label"].map(CATEGORY_LABELS).value_counts())

# --------------------------------------------------------------------------
# 4. Performa & Evaluasi Model Naive Bayes
# --------------------------------------------------------------------------

st.header("4. Performa & Evaluasi Model Naive Bayes")

with st.spinner("Melatih model..."):
    model_pipeline = train_model(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(label="🎯 Akurasi pada Data Uji", value=f"{akurasi * 100:.2f}%")
with col_m2:
    with st.spinner("Menghitung cross-validation..."):
        cv_scores = cross_validate(model_pipeline, X, y)
    st.metric(
        label="📊 Rata-rata Akurasi (5-Fold CV)",
        value=f"{cv_scores.mean() * 100:.2f}%",
        delta=f"± {cv_scores.std() * 100:.2f}%",
    )

col_eval1, col_eval2 = st.columns(2)

with col_eval1:
    st.subheader("Laporan Klasifikasi (Classification Report)")
    report_dict = classification_report(
        y_test, y_pred,
        target_names=[CATEGORY_LABELS[c] for c in sorted(CATEGORY_LABELS)],
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format(precision=2), width="stretch")

with col_eval2:
    st.subheader("Matriks Kebingungan (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_pred)
    labels_display = [CATEGORY_LABELS[c] for c in model_pipeline.classes_]

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
        xticklabels=labels_display,
        yticklabels=labels_display,
    )
    plt.ylabel("Aktual")
    plt.xlabel("Prediksi")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

# --------------------------------------------------------------------------
# Sidebar: Prediksi Sentimen Instan
# --------------------------------------------------------------------------

st.sidebar.header("Prediksi Sentimen Instan")
st.sidebar.markdown(
    "Masukkan ulasan baru di bawah ini untuk menguji performa model secara langsung."
)

user_input = st.sidebar.text_area(
    "Tulis Ulasan Anda:",
    placeholder="Contoh: aplikasinya sangat bagus dan cepat...",
)

if st.sidebar.button("Analisis Sentimen"):
    if user_input.strip() != "":
        cleaned_input = bersihkan_teks(user_input)

        if cleaned_input.strip() != "":
            proba = model_pipeline.predict_proba([cleaned_input])[0]
            nilai_numerik = int(model_pipeline.classes_[proba.argmax()])
            keyakinan = proba.max() * 100

            sentimen_teks = CATEGORY_LABELS.get(nilai_numerik, f"Label {nilai_numerik}")

            if nilai_numerik == 1:
                st.sidebar.success(f"Sentimen Terprediksi: **{sentimen_teks}** ({keyakinan:.1f}% keyakinan)")
            elif nilai_numerik == 0:
                st.sidebar.error(f"Sentimen Terprediksi: **{sentimen_teks}** ({keyakinan:.1f}% keyakinan)")
            else:
                st.sidebar.info(f"Sentimen Terprediksi: **{sentimen_teks}** ({keyakinan:.1f}% keyakinan)")
        else:
            st.sidebar.warning("Teks input tidak mengandung kata yang valid untuk dianalisis.")
    else:
        st.sidebar.warning("Silakan masukkan teks terlebih dahulu.")
