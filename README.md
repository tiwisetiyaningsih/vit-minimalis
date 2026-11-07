# Vision Transformer (ViT) Minimalis — Fashion-MNIST

Implementasi Vision Transformer (ViT) dari scratch menggunakan PyTorch untuk klasifikasi gambar Fashion-MNIST.

Proyek ini mencakup eksplorasi dataset, preprocessing, implementasi arsitektur Transformer, training, evaluasi, dan visualisasi hasil klasifikasi.

## 1. Hasil Proyek

Model baseline berhasil dilatih menggunakan CPU dan menghasilkan:

| Metrik | Hasil |
|---|---:|
| Best validation accuracy | 85.42% |
| Test accuracy | **85.97%** |
| Test loss | 0.3785 |
| Macro precision | 0.8601 |
| Macro recall | 0.8597 |
| Macro F1-score | 0.8585 |
| Trainable parameters | 72.074 |

Checkpoint terbaik diperoleh pada epoch ke-5.

## 2. Dataset

Dataset yang digunakan adalah Fashion-MNIST.

| Informasi | Train | Test |
|---|---:|---:|
| Jumlah gambar | 60.000 | 10.000 |
| Resolusi | 28 × 28 | 28 × 28 |
| Channel | Grayscale | Grayscale |
| Fitur piksel | 784 | 784 |
| Jumlah kelas | 10 | 10 |

Kelas yang digunakan:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Dataset mentah berada di folder `data/raw/` dan tidak dimodifikasi selama eksperimen.

### Catatan Audit Dataset

Audit menemukan:
- 43 kelompok duplikasi internal train.
- 1 kelompok duplikasi internal test.
- 10 kelompok duplikasi lintas train-test.
- 1 konflik label lintas train-test.

Duplikasi dipertahankan sesuai kebijakan kelompok. Duplikasi lintas train-test didokumentasikan sebagai potensi data leakage, sedangkan konflik label dicatat sebagai anomali. Label asli tidak diubah.

## 3. Arsitektur Vision Transformer

Model diimplementasikan secara manual menggunakan PyTorch.

| Komponen | Konfigurasi |
|---|---|
| Input | 28 × 28 grayscale |
| Patch size | 4 × 4 |
| Jumlah patch | 49 |
| Embedding dimension | 64 |
| Class token | 1 |
| Positional embedding | Learnable |
| Total token | 50 |
| Transformer blocks | 2 |
| Attention heads | 4 |
| Head dimension | 16 |
| Feed Forward Network | 64 → 128 → 64 |
| Output classes | 10 |

Komponen model:
- PatchEmbedding
- MultiHeadSelfAttention
- TransformerEncoderBlock
- VisionTransformer

Implementasi tidak menggunakan pretrained ViT, `nn.MultiheadAttention`, maupun `nn.TransformerEncoder`.

## 4. Preprocessing

Preprocessing dilakukan menggunakan `src/data/dataset.py`.

Tahapan:
1. Membaca CSV train dan test.
2. Memisahkan label dan fitur piksel.
3. Normalisasi nilai piksel 0–255 menjadi 0–1.
4. Mengubah gambar menjadi tensor `[N, 1, 28, 28]`.
5. Mengubah label menjadi tensor `int64`.
6. Membuat Dataset dan DataLoader.

Konfigurasi DataLoader:
- Batch size: 64.
- Image dtype: `float32`.
- Label dtype: `int64`.

## 5. Training

Training dilakukan menggunakan `src/training/train.py`.

| Parameter | Nilai |
|---|---|
| Training samples | 54.000 |
| Validation samples | 6.000 |
| Test samples | 10.000 |
| Epoch | 5 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Random seed | 42 |
| Device | CPU |

Validation set dibentuk dari dataset train menggunakan `random_split()`.

Test set tidak digunakan selama training maupun pemilihan checkpoint.

### Hasil Training

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.7606 | 70.96% | 0.5595 | 79.35% |
| 2 | 0.5223 | 80.70% | 0.4848 | 81.62% |
| 3 | 0.4732 | 82.49% | 0.4171 | 83.92% |
| 4 | 0.4409 | 83.74% | 0.4059 | 85.23% |
| 5 | 0.4248 | 84.17% | 0.3866 | 85.42% |

## 6. Evaluasi

Evaluasi akhir dilakukan menggunakan `src/evaluation/evaluate.py` dan checkpoint terbaik.

### Hasil Test

| Metrik | Nilai |
|---|---:|
| Test loss | 0.3785 |
| Test accuracy | 85.97% |
| Macro precision | 0.8601 |
| Macro recall | 0.8597 |
| Macro F1-score | 0.8585 |

### F1-score per Kelas

| Kelas | F1-score |
|---|---:|
| T-shirt/top | 0.82 |
| Trouser | 0.98 |
| Pullover | 0.76 |
| Dress | 0.87 |
| Coat | 0.76 |
| Sandal | 0.94 |
| Shirt | 0.65 |
| Sneaker | 0.91 |
| Bag | 0.96 |
| Ankle boot | 0.93 |

Trouser merupakan kelas dengan performa terbaik, sedangkan Shirt merupakan kelas dengan performa terendah.

## 7. Visualisasi dan Output

Hasil eksperimen disimpan dalam folder `outputs/`.

| File | Keterangan |
|---|---|
| `best_vit.pth` | Checkpoint model terbaik |
| `training_history.csv` | Riwayat training |
| `training_loss.png` | Grafik loss |
| `training_accuracy.png` | Grafik accuracy |
| `confusion_matrix.png` | Visualisasi confusion matrix |
| `evaluation_metrics.json` | Metrik evaluasi |
| `classification_report.txt` | Classification report |
| `classification_report.csv` | Classification report format CSV |
| `confusion_matrix.csv` | Confusion matrix format CSV |

Folder `outputs/` diabaikan oleh Git. Checkpoint dan hasil eksperimen dapat disimpan secara lokal tanpa ikut masuk commit.

## 8. Struktur Repository

```text
vit-minimalis/
├── data/
│   ├── raw/
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

## 9. Instalasi dan Penggunaan

### Prasyarat

- Python 3.11.
- PyTorch 2.15.0+cpu pada environment eksperimen.
- Git.
- Dataset Fashion-MNIST dalam format CSV.

### Clone Repository

```bash
git clone https://github.com/tiwisetiyaningsih/vit-minimalis.git
cd vit-minimalis
```

### Instalasi Dependensi

```bash
py -m pip install -r requirements.txt
```

Pastikan PyTorch tersedia dan gunakan environment yang kompatibel dengan versi eksperimen.

### Verifikasi PyTorch

```bash
py -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Menjalankan Preprocessing

```bash
py src/data/dataset.py
```

### Menguji Model ViT

```bash
py -m src.models.vit
```

### Menjalankan Training

```bash
py -m src.training.train
```

Training menghasilkan checkpoint dan riwayat training pada folder `outputs/`.

### Menjalankan Evaluasi

```bash
py -m src.evaluation.evaluate
```

Evaluasi menggunakan checkpoint terbaik dan test set untuk menghasilkan metrik serta visualisasi.

**Catatan:** Perintah training akan menjalankan training baru. Untuk melihat hasil eksperimen yang sudah selesai, gunakan file history dan checkpoint yang telah dihasilkan sebelumnya.

## 10. Keterbatasan

1. Eksperimen menggunakan satu konfigurasi baseline.
2. Belum dilakukan hyperparameter tuning.
3. Belum dilakukan perbandingan dengan arsitektur lain.
4. Training menggunakan CPU.
5. Performa kelas Shirt masih relatif rendah.
6. Terdapat potensi data leakage akibat duplikasi lintas train-test.
7. Terdapat satu konflik label pada gambar identik lintas train-test.

Hasil eksperimen belum dapat digunakan untuk menyatakan bahwa konfigurasi model merupakan konfigurasi optimal.

## 11. Dokumentasi Tambahan

Informasi lebih lanjut tersedia pada:
- `docs/experiment_log.md`
- `docs/debugging_log.md`
- `docs/llm_interaction_log.md`
- `docs/project_planning.md`
- `docs/contribution_log.md`

## 12. Kesimpulan

Proyek berhasil mengimplementasikan Vision Transformer minimalis dari scratch menggunakan PyTorch untuk klasifikasi Fashion-MNIST.

Model menghasilkan test accuracy sebesar 85.97% dan macro F1-score sebesar 0.8585.

Seluruh tahapan utama mulai dari eksplorasi dataset, preprocessing, implementasi model, training, evaluasi, hingga visualisasi telah diselesaikan.