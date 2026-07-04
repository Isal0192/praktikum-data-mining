# Aplikasi Analisis Sentimen Ulasan E-Commerce
 
## Kelompok

 - Faisal Fajar
 - Muhamad Zulfikar
 - IRFAN GUNAWAN PRATAMA

## Tentang Dataset
 
**File:** `data_real.csv`
 
Dataset berisi ulasan (review) produk dari platform e-commerce, dengan atribut:
 
- **Ulasan** - teks ulasan yang dituliskan konsumen
- **Rating** - skor rating yang diberikan konsumen (1-5)
- **Kategori** - kategori produk (pertukangan, handphone, fashion, elektronik, olahraga)
- **Nama Produk, Id Produk, Terjual, Id_Toko, Url** - atribut deskriptif produk dan toko
- **label** - label sentimen target (`0` = Negatif, `1` = Positif)
Total data: 1.925 baris (sebelum pembersihan), 1.904 baris siap dianalisis setelah pra-pemrosesan.
 
## Tujuan
 
1. Mengklasifikasikan sentimen ulasan produk e-commerce (**positif**/**negatif**) menggunakan algoritma **Naive Bayes**.
2. Menerapkan pra-pemrosesan teks berbahasa Indonesia (normalisasi negasi & singkatan, pembersihan stopword) agar teks siap diolah model machine learning.
3. Mengevaluasi performa model menggunakan akurasi, precision, recall, F1-score, confusion matrix, dan cross-validation.
4. Menyediakan visualisasi data (distribusi label, word cloud) serta fitur **prediksi sentimen interaktif** melalui aplikasi berbasis Streamlit.
