# Project Planning — Vision Transformer Minimalis

## 1. Informasi Proyek

| Keterangan | Detail |
|---|---|
| Nama Proyek | Vision Transformer (ViT) Minimalis |
| Dataset | Fashion-MNIST |
| Framework | PyTorch |
| Bahasa | Python |
| Eksekusi | CPU |
| Status | Implementasi, training, evaluasi, dan dokumentasi final |

## 2. Tujuan Proyek

Mengimplementasikan model Vision Transformer minimalis dari scratch untuk klasifikasi gambar Fashion-MNIST.

Proyek mencakup eksplorasi dataset, preprocessing, implementasi arsitektur, training, evaluasi, visualisasi, dan dokumentasi.

## 3. Ruang Lingkup

### 3.1 Dataset

- Membaca Fashion-MNIST dalam format CSV.
- Memeriksa karakteristik dan distribusi kelas.
- Melakukan audit duplikasi.
- Mempertahankan dataset mentah.

### 3.2 Preprocessing

- Memisahkan label dan fitur piksel.
- Normalisasi piksel menjadi 0–1.
- Mengubah gambar menjadi tensor `[N, 1, 28, 28]`.
- Membuat Dataset dan DataLoader.

### 3.3 Implementasi Model

Membangun komponen:
- Patch Embedding.
- Multi-Head Self-Attention.
- Transformer Encoder Block.
- Class Token.
- Positional Embedding.
- Classification Head.

### 3.4 Training dan Evaluasi

- Membagi data train menjadi training dan validation.
- Melatih model menggunakan Adam dan CrossEntropyLoss.
- Menyimpan checkpoint berdasarkan validation accuracy.
- Melakukan evaluasi akhir menggunakan test set.
- Menghasilkan classification report dan confusion matrix.

### 3.5 Dokumentasi

- Experiment log.
- Debugging log.
- LLM interaction log.
- Project planning.
- README.

## 4. Rencana Tahapan dan Status

| Tahap | Aktivitas | Status |
|---|---|---|
| 1 | Dataset Exploration | Selesai |
| 2 | Duplicate Audit | Selesai |
| 3 | Preprocessing PyTorch | Selesai |
| 4 | Implementasi Model ViT | Selesai |
| 5 | Forward Pass Testing | Selesai |
| 6 | Training Baseline | Selesai |
| 7 | Validation | Selesai |
| 8 | Test Evaluation | Selesai |
| 9 | Visualisasi | Selesai |
| 10 | Dokumentasi | Finalisasi |

## 5. Konfigurasi Model

| Parameter | Nilai |
|---|---:|
| Input image | 28 × 28 |
| Channel | 1 |
| Patch size | 4 × 4 |
| Jumlah patch | 49 |
| Embedding dimension | 64 |
| Class token | 1 |
| Positional embedding | Learnable |
| Transformer blocks | 2 |
| Attention heads | 4 |
| Head dimension | 16 |
| MLP | 64 → 128 → 64 |
| Output classes | 10 |
| Trainable parameters | 72.074 |

## 6. Konfigurasi Training

| Parameter | Nilai |
|---|---|
| Training data | 54.000 |
| Validation data | 6.000 |
| Test data | 10.000 |
| Batch size | 64 |
| Epoch | 5 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Random seed | 42 |
| Device | CPU |

## 7. Target dan Hasil Aktual

| Metrik | Hasil |
|---|---:|
| Best validation accuracy | 85.42% |
| Test accuracy | 85.97% |
| Test loss | 0.3785 |
| Macro precision | 0.8601 |
| Macro recall | 0.8597 |
| Macro F1-score | 0.8585 |

Kelas dengan performa terbaik adalah Trouser dengan F1-score 0.98.

Kelas dengan performa terendah adalah Shirt dengan F1-score 0.65.

## 8. Output Proyek

### Source Code

```text
src/
├── data/
│   └── dataset.py
├── models/
│   └── vit.py
├── training/
│   └── train.py
└── evaluation/
    └── evaluate.py
```

### Dokumentasi

```text
docs/
├── contribution_log.md
├── debugging_log.md
├── experiment_log.md
├── llm_interaction_log.md
└── project_planning.md
```

### Hasil Eksperimen

```text
outputs/
├── best_vit.pth
├── training_history.csv
├── training_loss.png
├── training_accuracy.png
├── confusion_matrix.png
├── evaluation_metrics.json
├── classification_report.txt
├── classification_report.csv
└── confusion_matrix.csv
```

## 9. Risiko dan Keterbatasan

1. Training dilakukan menggunakan CPU.
2. Eksperimen hanya mencakup satu konfigurasi baseline.
3. Belum dilakukan hyperparameter tuning.
4. Belum dilakukan eksperimen pembanding.
5. Beberapa kelas, terutama Shirt, memiliki performa lebih rendah.
6. Terdapat duplikasi lintas train-test yang berpotensi menyebabkan data leakage.
7. Terdapat satu anomali label lintas train-test.

Dataset mentah dipertahankan sesuai kebijakan kelompok.

## 10. Pekerjaan Finalisasi

Pekerjaan finalisasi repository meliputi:

- [x] Menyelesaikan preprocessing.
- [x] Menyelesaikan implementasi model.
- [x] Menyelesaikan training.
- [x] Menyelesaikan evaluasi.
- [x] Menghasilkan visualisasi.
- [x] Menyimpan checkpoint dan training history.
- [x] Menyusun dokumentasi eksperimen.
- [ ] Memeriksa kelengkapan seluruh file dokumentasi.
- [ ] Memeriksa `.gitignore` dan `requirements.txt`.
- [ ] Memeriksa status Git.
- [ ] Commit perubahan final.
- [ ] Push ke GitHub branch `main`.

## 11. Status Akhir

Implementasi baseline Vision Transformer untuk klasifikasi Fashion-MNIST telah selesai dan berhasil dievaluasi.

Tahap yang tersisa adalah pemeriksaan akhir dokumentasi dan sinkronisasi perubahan ke repository GitHub.