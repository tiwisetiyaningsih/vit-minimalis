# Debugging Log — Vision Transformer Minimalis

## 1. Informasi

Dokumen ini mencatat kendala teknis yang ditemukan selama pengembangan proyek ViT Minimalis dan tindakan penyelesaiannya.

## 2. Daftar Kendala dan Solusi

### 2.1 Variabel Audit Belum Didefinisikan

**Tahap:** Dataset Exploration

**Gejala:**

Kode audit duplikasi membutuhkan variabel nama kolom label dan fitur piksel yang belum didefinisikan pada bagian sebelumnya.

**Penyelesaian:**

Menambahkan cell persiapan sebelum kode audit untuk mendefinisikan:
- `train_label_col`
- `test_label_col`
- `train_pixel_columns`
- `test_pixel_columns`

Jumlah fitur piksel juga diperiksa agar sesuai dengan 784 kolom.

**Status:** Selesai.

### 2.2 Warning NumPy Non-Writable Array

**Tahap:** Preprocessing

**Gejala:**

PyTorch menampilkan warning ketika array NumPy yang tidak writable diteruskan ke `torch.from_numpy()`.

Warning muncul pada konversi label.

**Penyelesaian:**

Konversi label diubah menjadi:

```python
label_tensor = torch.tensor(labels, dtype=torch.int64)
```

Pendekatan tersebut membuat tensor baru dan menghindari penggunaan array non-writable secara langsung.

**Hasil:**

Preprocessing dijalankan kembali dan warning tidak muncul lagi.

**Status:** Selesai.

### 2.3 Import Module Python

**Tahap:** Implementasi Model dan Training

**Gejala:**

Import modul dalam struktur package `src` memerlukan eksekusi dari root repository agar path package dikenali dengan benar.

**Penyelesaian:**

Menambahkan file `__init__.py` pada package yang diperlukan dan menggunakan eksekusi berbasis module dari root repository.

Contoh:

```bash
py -m src.models.vit
```

Untuk training:

```bash
py -m src.training.train
```

Untuk evaluasi:

```bash
py -m src.evaluation.evaluate
```

**Status:** Selesai.

### 2.4 Kesalahan Penulisan Nama File Evaluasi

**Tahap:** Evaluasi

**Gejala:**

Nama file evaluasi mengalami kesalahan penulisan antara `evaluate.py` dan `evaluate,py`.

**Penyelesaian:**

Menggunakan nama file Python yang benar:

```text
src/evaluation/evaluate.py
```

Script kemudian dijalankan melalui:

```bash
py -m src.evaluation.evaluate
```

**Status:** Selesai.

### 2.5 Instalasi scikit-learn

**Tahap:** Evaluasi

**Gejala:**

Library scikit-learn diperlukan untuk menghasilkan classification report, confusion matrix, precision, recall, dan F1-score.

**Penyelesaian:**

Memastikan scikit-learn tersedia pada environment Python yang digunakan.

Jika diperlukan instalasi melalui terminal:

```bash
py -m pip install scikit-learn
```

Dependensi juga dicantumkan dalam `requirements.txt`:

```text
scikit-learn
```

**Status:** Dependensi digunakan dalam evaluasi.

### 2.6 Validasi Bentuk Tensor

**Tahap:** Preprocessing dan Implementasi Model

**Gejala:**

Model membutuhkan input gambar berbentuk empat dimensi dengan satu channel.

**Penyelesaian:**

Memastikan preprocessing menghasilkan:

```text
[N, 1, 28, 28]
```

Forward pass kemudian diuji menggunakan satu batch berukuran 64.

**Hasil:**

```text
Input shape           : [64, 1, 28, 28]
Patch embedding shape : [64, 49, 64]
Tokens shape          : [64, 50, 64]
Output shape          : [64, 10]
```

Tidak ditemukan NaN maupun Inf.

**Status:** Selesai.

## 3. Ringkasan

| No. | Kendala | Penyelesaian | Status |
|---:|---|---|---|
| 1 | Variabel audit belum didefinisikan | Menambahkan cell persiapan | Selesai |
| 2 | Warning NumPy non-writable | Menggunakan `torch.tensor()` | Selesai |
| 3 | Import module | Menambahkan package initialization dan menjalankan dengan `-m` | Selesai |
| 4 | Salah penulisan `evaluate.py` | Menggunakan nama file yang benar | Selesai |
| 5 | Kebutuhan scikit-learn | Memastikan dependensi tersedia | Selesai |
| 6 | Validasi bentuk tensor | Memperbaiki dan memvalidasi pipeline tensor | Selesai |

## 4. Catatan

Dokumen ini mencatat kendala yang diketahui selama proses pengembangan. Pesan traceback asli dan waktu kejadian tidak dicantumkan apabila tidak tersedia.

Seluruh kendala yang tercatat tidak lagi menghalangi eksekusi preprocessing, forward pass, training, maupun evaluasi model.