# LLM Interaction Log — Vision Transformer Minimalis

## 1. Informasi

Dokumen ini mencatat penggunaan Large Language Model (LLM) sebagai alat bantu selama proses pengembangan proyek Vision Transformer Minimalis.

LLM digunakan sebagai pendamping teknis untuk membantu memahami konsep, menyusun kode, melakukan debugging, memeriksa hasil eksperimen, dan menyusun dokumentasi.

## 2. Ruang Lingkup Penggunaan LLM

### 2.1 Dataset Exploration

**Aktivitas:**
- Membantu menyusun alur eksplorasi Fashion-MNIST.
- Membantu memeriksa struktur dataset.
- Membantu menginterpretasikan distribusi kelas dan karakteristik piksel.
- Membantu memahami hasil audit duplikasi.

**Hasil:**

Eksplorasi dataset diselesaikan dan hasilnya dicatat dalam notebook serta experiment log.

### 2.2 Duplicate Audit

**Aktivitas:**
- Membantu memahami metode identifikasi gambar duplikat.
- Membantu memeriksa konsistensi label.
- Membantu menjelaskan potensi data leakage akibat gambar identik pada train dan test.

**Hasil:**

Ditemukan duplikasi internal dan lintas dataset serta satu konflik label lintas train-test.

Kebijakan penanganan duplikasi ditetapkan oleh kelompok. Dataset mentah tetap dipertahankan.

### 2.3 Preprocessing dengan PyTorch

**Aktivitas:**
- Membantu merancang pembacaan CSV.
- Membantu memisahkan label dan fitur piksel.
- Membantu membuat tensor gambar.
- Membantu merancang Dataset dan DataLoader.
- Membantu memperbaiki warning NumPy non-writable array.

**Hasil:**

Preprocessing berhasil menghasilkan tensor gambar `[N, 1, 28, 28]`, label `int64`, dan DataLoader dengan batch size 64.

### 2.4 Implementasi Vision Transformer

**Aktivitas:**
- Membantu menyusun komponen Patch Embedding.
- Membantu memahami Scaled Dot-Product Attention.
- Membantu merancang Multi-Head Self-Attention manual.
- Membantu menyusun Transformer Encoder Block.
- Membantu mengintegrasikan class token, positional embedding, dan classification head.
- Membantu memvalidasi bentuk output model.

**Hasil:**

Model ViT berhasil diimplementasikan dari scratch dan menghasilkan output `[64, 10]` pada forward pass.

### 2.5 Training Model

**Aktivitas:**
- Membantu menyusun training loop.
- Membantu membuat validation loop.
- Membantu menerapkan random split dengan seed tetap.
- Membantu menyimpan checkpoint terbaik.
- Membantu mencatat training history ke CSV.

**Hasil:**

Training baseline berhasil diselesaikan dalam lima epoch dengan best validation accuracy 85.42%.

### 2.6 Evaluasi dan Visualisasi

**Aktivitas:**
- Membantu menyusun script evaluasi terpisah.
- Membantu menggunakan checkpoint terbaik.
- Membantu menghitung classification metrics.
- Membantu menghasilkan confusion matrix.
- Membantu membuat grafik training dan validation.

**Hasil:**

Evaluasi test menghasilkan accuracy 85.97% dan macro F1-score 0.8585.

### 2.7 Debugging

LLM digunakan untuk membantu menganalisis kendala teknis, antara lain:
- Variabel audit yang belum didefinisikan.
- Warning NumPy non-writable array.
- Import module pada struktur package Python.
- Kesalahan penulisan nama file evaluasi.
- Kebutuhan dependensi scikit-learn.

Solusi diperiksa melalui eksekusi ulang oleh pengguna.

### 2.8 Dokumentasi

LLM membantu menyusun dan memperbarui:
- Experiment log.
- Debugging log.
- LLM interaction log.
- Project planning.
- README.

Informasi dokumentasi disesuaikan dengan hasil eksekusi dan metrik aktual yang telah dikonfirmasi.

## 3. Peran LLM dan Pengguna

| Aktivitas | Peran LLM | Peran Pengguna |
|---|---|---|
| Eksplorasi dataset | Membantu menyusun alur dan interpretasi | Menjalankan notebook dan memeriksa hasil |
| Audit duplikasi | Membantu analisis dan penjelasan | Memverifikasi hasil audit dan mengikuti keputusan kelompok |
| Preprocessing | Membantu menyusun kode dan debugging | Mengimplementasikan dan menguji kode |
| Model ViT | Membantu merancang komponen arsitektur | Menjalankan forward pass |
| Training | Membantu menyusun pipeline training | Menjalankan training dan memverifikasi hasil |
| Evaluasi | Membantu menyusun metrik dan visualisasi | Menjalankan evaluasi |
| Dokumentasi | Membantu menyusun isi dokumen | Memeriksa dan menyesuaikan dokumen proyek |

## 4. Batasan Penggunaan LLM

1. Kode yang dihasilkan perlu diperiksa dan diuji sebelum digunakan.
2. Hasil eksperimen hanya dicatat setelah eksekusi berhasil dikonfirmasi.
3. LLM tidak digunakan untuk mengarang hasil training atau evaluasi.
4. LLM tidak digunakan untuk mengubah hasil eksperimen yang sudah diperoleh.
5. Keputusan kelompok mengenai penanganan duplikasi tetap menjadi keputusan anggota kelompok.
6. Penggunaan LLM tidak menggantikan proses verifikasi teknis oleh pengguna.

## 5. Kesimpulan

LLM digunakan sebagai alat bantu dalam proses pengembangan ViT Minimalis, khususnya untuk penyusunan kode, penjelasan konsep, debugging, interpretasi hasil, dan dokumentasi.

Seluruh hasil yang dicantumkan dalam laporan didasarkan pada eksekusi dan hasil aktual proyek yang telah dikonfirmasi.