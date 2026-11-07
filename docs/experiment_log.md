# Experiment Log — Vision Transformer (ViT) Minimalis

## 1. Informasi Proyek

| Keterangan | Detail |
|---|---|
| Nama Proyek | Vision Transformer (ViT) Minimalis |
| Dataset | Fashion-MNIST |
| Framework | PyTorch |
| Bahasa Pemrograman | Python |
| Lingkungan Pengembangan | VS Code dan Jupyter Notebook |
| Perangkat Eksekusi | CPU |
| Versi PyTorch | 2.15.0+cpu |
| Status | Eksplorasi, preprocessing, implementasi model, training, dan evaluasi selesai |

## 2. Tujuan Eksperimen

Proyek ini bertujuan mengimplementasikan arsitektur Vision Transformer (ViT) secara manual untuk melakukan klasifikasi gambar Fashion-MNIST.

Eksperimen mencakup:
1. Eksplorasi dan pemahaman karakteristik dataset.
2. Audit kualitas dan duplikasi data.
3. Preprocessing gambar menggunakan PyTorch.
4. Implementasi arsitektur Vision Transformer dari scratch.
5. Training model menggunakan data training.
6. Validasi model menggunakan validation set.
7. Evaluasi akhir menggunakan test set.
8. Analisis hasil klasifikasi dan keterbatasan model.

Implementasi model dilakukan tanpa menggunakan pretrained ViT, `nn.MultiheadAttention`, maupun `nn.TransformerEncoder`.

## 3. Informasi Dataset

Dataset yang digunakan adalah Fashion-MNIST dengan dua file CSV:

- `data/raw/fashion-mnist_train.csv`
- `data/raw/fashion-mnist_test.csv`

### 3.1 Karakteristik Dataset

| Karakteristik | Train | Test |
|---|---:|---:|
| Jumlah gambar | 60.000 | 10.000 |
| Jumlah kolom | 785 | 785 |
| Jumlah fitur piksel | 784 | 784 |
| Resolusi gambar | 28 × 28 | 28 × 28 |
| Channel | Grayscale | Grayscale |
| Tipe data awal | int64 | int64 |
| Missing values | 0 | 0 |

Setiap baris dataset terdiri dari satu kolom label dan 784 kolom fitur piksel.

Nilai piksel berada pada rentang 0–255.

### 3.2 Kelas Fashion-MNIST

Dataset memiliki 10 kelas:

| Label | Nama Kelas |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Distribusi kelas seimbang:
- Train: 6.000 gambar per kelas.
- Test: 1.000 gambar per kelas.

### 3.3 Hasil Eksplorasi

Hasil eksplorasi dataset menunjukkan:
- Tidak ditemukan missing values.
- Seluruh fitur piksel menggunakan tipe data `int64`.
- Nilai piksel berada pada rentang 0–255.
- Rata-rata nilai piksel train adalah 72.956831.
- Rata-rata nilai piksel test adalah 73.160805.
- Gambar berhasil direkonstruksi menjadi bentuk 28 × 28 grayscale.
- Distribusi kelas train dan test seimbang.

## 4. Audit Duplikasi Dataset

Audit dilakukan terhadap fitur piksel untuk mengidentifikasi gambar identik di dalam train, di dalam test, dan lintas train-test.

### 4.1 Hasil Audit

| Temuan | Hasil |
|---|---:|
| Kelompok duplikasi internal train | 43 |
| Kelompok duplikasi internal test | 1 |
| Kelompok duplikasi lintas train-test | 10 |
| Konflik label internal train | 0 |
| Konflik label internal test | 0 |
| Konflik label lintas train-test | 1 |

### 4.2 Konflik Label

Ditemukan satu gambar yang memiliki seluruh 784 nilai piksel identik, tetapi memiliki label berbeda pada train dan test.

| Dataset | Indeks Posisi | Label |
|---|---:|---|
| Train | 52205 | 6 — Shirt |
| Test | 9560 | 0 — T-shirt/top |

Perbandingan langsung terhadap seluruh 784 fitur piksel menghasilkan nilai `True`, dengan jumlah piksel berbeda sebanyak 0.

### 4.3 Kebijakan Penanganan Duplikasi

Kelompok menetapkan kebijakan berikut:

1. Duplikasi internal train sebanyak 43 kelompok dipertahankan.
2. Duplikasi internal test sebanyak 1 kelompok dipertahankan.
3. Duplikasi lintas train-test sebanyak 10 kelompok dipertahankan dan didokumentasikan sebagai potensi data leakage.
4. Satu konflik label lintas train-test didokumentasikan sebagai anomali.
5. Label asli tidak diubah.
6. Tidak ada baris yang dihapus dari dataset mentah.
7. Tidak ada file CSV mentah yang dimodifikasi.

Potensi data leakage dicatat sebagai keterbatasan dataset. Besarnya pengaruh duplikasi terhadap performa model tidak diukur secara terpisah.

## 5. Preprocessing Fashion-MNIST

Preprocessing diimplementasikan menggunakan PyTorch pada:

`src/data/dataset.py`

### 5.1 Tahapan Preprocessing

1. Membaca CSV train dan test.
2. Memisahkan kolom `label` dan 784 fitur piksel.
3. Memastikan urutan fitur train dan test konsisten.
4. Memvalidasi missing values dan rentang piksel.
5. Melakukan normalisasi nilai piksel.
6. Mengubah gambar menjadi tensor.
7. Membuat class `FashionMNISTDataset`.
8. Membuat DataLoader.

### 5.2 Normalisasi

Nilai piksel dinormalisasi menggunakan:

`pixel_normalized = pixel / 255.0`

Dengan demikian, rentang nilai piksel berubah dari 0–255 menjadi 0–1.

### 5.3 Bentuk Tensor

| Komponen | Bentuk | Tipe Data |
|---|---|---|
| Train images | [60000, 1, 28, 28] | float32 |
| Test images | [10000, 1, 28, 28] | float32 |
| Train labels | [60000] | int64 |
| Test labels | [10000] | int64 |
| Batch images | [64, 1, 28, 28] | float32 |
| Batch labels | [64] | int64 |

Preprocessing berhasil divalidasi dengan rentang piksel 0–1 dan seluruh label berada pada rentang 0–9.

Warning NumPy terkait array non-writable diperbaiki dengan membuat tensor label melalui `torch.tensor(labels, dtype=torch.int64)`.

Hasil akhir:

`PREPROCESSING VALIDATION PASSED!`

## 6. Implementasi Model Vision Transformer

Implementasi model terdapat pada:

`src/models/vit.py`

Model dibuat dari scratch menggunakan PyTorch.

### 6.1 Konfigurasi Arsitektur

| Komponen | Nilai |
|---|---:|
| Image size | 28 × 28 |
| Channel | 1 |
| Patch size | 4 × 4 |
| Jumlah patch | 49 |
| Embedding dimension | 64 |
| Class token | 1 |
| Positional embedding | Learnable |
| Total token | 50 |
| Transformer encoder blocks | 2 |
| Attention heads | 4 |
| Dimension per head | 16 |
| Feed Forward Network | 64 → 128 → 64 |
| Output classes | 10 |
| Trainable parameters | 72.074 |

### 6.2 Komponen Model

Model terdiri atas:

1. **PatchEmbedding**
   Membagi gambar menjadi 49 patch berukuran 4 × 4 dan mengubah setiap patch menjadi embedding 64 dimensi.

2. **MultiHeadSelfAttention**
   Mengimplementasikan Query, Key, Value, Scaled Dot-Product Attention, dan Softmax secara manual.

3. **TransformerEncoderBlock**
   Menggunakan Layer Normalization, Multi-Head Self-Attention, residual connection, dan Feed Forward Network.

4. **Class Token dan Positional Embedding**
   Menggunakan satu class token dan learnable positional embedding dengan panjang 50 token.

5. **Classification Head**
   Menggunakan representasi class token untuk menghasilkan logits dari 10 kelas.

### 6.3 Pengujian Forward Pass

Pengujian dilakukan menggunakan satu batch dari DataLoader.

| Tahap | Bentuk Tensor |
|---|---|
| Input | [64, 1, 28, 28] |
| Patch embedding | [64, 49, 64] |
| Setelah class token dan positional embedding | [64, 50, 64] |
| Output classification head | [64, 10] |

Pengujian menunjukkan:
- Patch embedding berhasil.
- Class token berhasil.
- Positional embedding berhasil.
- Transformer berhasil.
- Classification head berhasil.
- Tidak ditemukan NaN atau Inf.
- Model berjalan menggunakan CPU.

Hasil:

`VIT FORWARD PASS TEST PASSED!`

## 7. Training Model

Training dilakukan menggunakan:

`src/training/train.py`

### 7.1 Pembagian Dataset

| Data | Jumlah |
|---|---:|
| Training | 54.000 |
| Validation | 6.000 |
| Test | 10.000 |

Validation set diperoleh dari dataset train menggunakan `torch.utils.data.random_split()` dengan random seed 42.

Test set tidak digunakan selama training maupun pemilihan checkpoint.

### 7.2 Konfigurasi Training

| Parameter | Nilai |
|---|---|
| Device | CPU |
| Epoch | 5 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Random seed | 42 |

### 7.3 Hasil Training

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.7606 | 70.96% | 0.5595 | 79.35% |
| 2 | 0.5223 | 80.70% | 0.4848 | 81.62% |
| 3 | 0.4732 | 82.49% | 0.4171 | 83.92% |
| 4 | 0.4409 | 83.74% | 0.4059 | 85.23% |
| 5 | 0.4248 | 84.17% | 0.3866 | 85.42% |

Checkpoint terbaik diperoleh pada epoch 5 dengan validation accuracy 85.42%.

File yang dihasilkan:
- `outputs/best_vit.pth`
- `outputs/training_history.csv`

Training selesai menggunakan CPU. Tidak dilakukan training ulang setelah checkpoint terbaik berhasil disimpan.

## 8. Evaluasi Model

Evaluasi akhir dilakukan menggunakan:

`src/evaluation/evaluate.py`

Checkpoint yang digunakan:

`outputs/best_vit.pth`

Seluruh 10.000 data test digunakan untuk evaluasi akhir setelah konfigurasi baseline ditetapkan.

Test set tidak digunakan untuk tuning hyperparameter.

### 8.1 Hasil Evaluasi Keseluruhan

| Metrik | Hasil |
|---|---:|
| Test samples | 10.000 |
| Test loss | 0.3785 |
| Test accuracy | 85.97% |
| Macro precision | 0.8601 |
| Macro recall | 0.8597 |
| Macro F1-score | 0.8585 |

### 8.2 Classification Report

| Kelas | Precision | Recall | F1-score |
|---|---:|---:|---:|
| T-shirt/top | 0.81 | 0.83 | 0.82 |
| Trouser | 0.98 | 0.97 | 0.98 |
| Pullover | 0.71 | 0.81 | 0.76 |
| Dress | 0.83 | 0.92 | 0.87 |
| Coat | 0.77 | 0.74 | 0.76 |
| Sandal | 0.98 | 0.91 | 0.94 |
| Shirt | 0.71 | 0.59 | 0.65 |
| Sneaker | 0.89 | 0.94 | 0.91 |
| Bag | 0.98 | 0.95 | 0.96 |
| Ankle boot | 0.93 | 0.94 | 0.93 |

### 8.3 Analisis Hasil

Kelas dengan F1-score tertinggi adalah Trouser dengan nilai 0.98.

Kelas dengan F1-score terendah adalah Shirt dengan nilai 0.65. Recall Shirt sebesar 0.59 menunjukkan bahwa model masih kesulitan mengenali sebagian gambar dari kelas tersebut.

Performa kelas seperti Trouser, Bag, dan Sandal relatif tinggi, sedangkan Pullover, Coat, dan Shirt masih menunjukkan ruang perbaikan.

## 9. Analisis Perkembangan Training

Selama lima epoch:
- Training accuracy meningkat dari 70.96% menjadi 84.17%.
- Validation accuracy meningkat dari 79.35% menjadi 85.42%.
- Training loss menurun dari 0.7606 menjadi 0.4248.
- Validation loss menurun dari 0.5595 menjadi 0.3866.

Kedua loss terus menurun dan validation accuracy terus meningkat selama eksperimen.

Berdasarkan hasil tersebut, belum terlihat indikasi kuat overfitting selama lima epoch training.

Belum cukup bukti untuk menyimpulkan underfitting secara pasti tanpa eksperimen pembanding atau target performa tertentu.

Test accuracy sebesar 85.97% sedikit lebih tinggi daripada validation accuracy sebesar 85.42%.

### 9.1 Eksperimen Pembanding dan Hyperparameter Tuning

Eksperimen yang telah dilakukan merupakan baseline ViT dengan satu konfigurasi:

- Patch size 4.
- Embedding dimension 64.
- Dua transformer encoder blocks.
- Empat attention heads.
- MLP 64 → 128 → 64.
- Lima epoch.
- Learning rate 0.001.
- Batch size 64.

Belum dilakukan eksperimen pembanding antararsitektur maupun hyperparameter tuning. Oleh karena itu, hasil yang diperoleh merupakan hasil baseline dan tidak digunakan untuk menyimpulkan konfigurasi ini sebagai konfigurasi optimal.

## 10. Visualisasi

Script evaluasi menghasilkan visualisasi berikut:

| File | Keterangan |
|---|---|
| `outputs/training_loss.png` | Grafik training loss dan validation loss |
| `outputs/training_accuracy.png` | Grafik training accuracy dan validation accuracy |
| `outputs/confusion_matrix.png` | Visualisasi confusion matrix |

File pendukung:
- `outputs/evaluation_metrics.json`
- `outputs/classification_report.txt`
- `outputs/classification_report.csv`
- `outputs/confusion_matrix.csv`

## 11. Keterbatasan Eksperimen

Beberapa keterbatasan yang perlu diperhatikan:

1. Model hanya diuji menggunakan satu konfigurasi baseline.
2. Belum dilakukan hyperparameter tuning.
3. Belum dilakukan perbandingan dengan arsitektur klasifikasi lain.
4. Training menggunakan CPU.
5. Kinerja klasifikasi kelas Shirt masih lebih rendah dibandingkan kelas lainnya.
6. Dataset memiliki duplikasi lintas train-test yang berpotensi menyebabkan data leakage.
7. Ditemukan satu gambar identik dengan label berbeda antara train dan test.
8. Dampak kuantitatif duplikasi dan konflik label terhadap hasil evaluasi belum diukur secara terpisah.

Dataset mentah dipertahankan sesuai kebijakan kelompok. Tidak ada penghapusan gambar maupun perubahan label.

## 12. Struktur File Proyek

```text
vit-minimalis/
├── data/
│   ├── raw/
│   │   ├── fashion-mnist_train.csv
│   │   └── fashion-mnist_test.csv
│   └── processed/
├── docs/
│   ├── contribution_log.md
│   ├── debugging_log.md
│   ├── experiment_log.md
│   ├── llm_interaction_log.md
│   └── project_planning.md
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── outputs/
│   ├── best_vit.pth
│   ├── training_history.csv
│   ├── training_loss.png
│   ├── training_accuracy.png
│   ├── confusion_matrix.png
│   ├── evaluation_metrics.json
│   ├── classification_report.txt
│   ├── classification_report.csv
│   └── confusion_matrix.csv
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   └── vit.py
│   ├── training/
│   │   └── train.py
│   └── evaluation/
│       └── evaluate.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## 13. Status Akhir Eksperimen

| Tahapan | Status |
|---|---|
| Dataset Exploration | Selesai |
| Duplicate Audit | Selesai |
| Preprocessing | Selesai |
| Model Implementation | Selesai |
| Forward Pass Testing | Berhasil |
| Training | Selesai |
| Validation | Selesai |
| Test Evaluation | Selesai |
| Visualization | Selesai |
| Experiment Documentation | Diperbarui |

## 14. Kesimpulan

Model Vision Transformer minimalis berhasil diimplementasikan dari scratch menggunakan PyTorch dan dilatih pada dataset Fashion-MNIST.

Model menghasilkan test accuracy sebesar 85.97%, macro precision 0.8601, macro recall 0.8597, dan macro F1-score 0.8585.

Model memperoleh performa terbaik pada kelas Trouser dengan F1-score 0.98, sedangkan kelas Shirt memiliki F1-score terendah sebesar 0.65.

Eksperimen berhasil menyelesaikan rangkaian eksplorasi dataset, preprocessing, implementasi model, training, validasi, evaluasi, dan visualisasi. Hasil ini merupakan baseline yang masih dapat dikembangkan melalui eksperimen lanjutan.