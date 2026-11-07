# Experiment Log — Vision Transformer (ViT) Minimalis

## 1. Informasi Proyek

| Keterangan | Detail |
|---|---|
| Nama Proyek | Vision Transformer (ViT) Minimalis |
| Dataset | Fashion-MNIST |
| Tahap | Dataset Exploration |
| Bahasa Pemrograman | Python |
| Library Utama | Pandas, NumPy, Matplotlib |
| Lingkungan Pengembangan | VS Code Jupyter Notebook |
| Notebook | `notebooks/01_dataset_exploration.ipynb` |
| Status | Eksplorasi selesai, evaluasi temuan menunggu keputusan kelompok |

## 2. Tujuan Eksperimen

Dataset Exploration dilakukan untuk memahami karakteristik dan kualitas dataset Fashion-MNIST sebelum digunakan dalam pengembangan model Vision Transformer (ViT) Minimalis.

Pemeriksaan meliputi dimensi dataset, struktur kolom, tipe data, missing values, duplikasi, distribusi label, statistik nilai piksel, serta visualisasi gambar.

Audit tambahan dilakukan untuk memeriksa gambar identik berdasarkan 784 fitur piksel, termasuk kemungkinan gambar yang sama muncul pada data train dan test dengan label berbeda.

## 3. Informasi Dataset

Dataset terdiri dari dua file:

- `data/raw/fashion-mnist_train.csv`
- `data/raw/fashion-mnist_test.csv`

Hasil pemeriksaan awal:

| Keterangan | Train | Test |
|---|---:|---:|
| Jumlah sampel | 60.000 | 10.000 |
| Jumlah kolom | 785 | 785 |
| Kolom label | 1 | 1 |
| Fitur piksel | 784 | 784 |
| Missing values | 0 | 0 |
| Duplikasi seluruh baris | 43 | 1 |

Seluruh kolom memiliki tipe data `int64`.

## 4. Hasil Eksplorasi

### 4.1 Struktur Dataset

Setiap baris merepresentasikan satu gambar Fashion-MNIST. Dataset memiliki satu kolom label dan 784 kolom fitur piksel.

Jumlah fitur piksel sesuai dengan ukuran gambar 28 × 28 piksel dengan satu channel grayscale.

### 4.2 Distribusi Label

Dataset memiliki 10 kelas dengan distribusi seimbang.

| Label | Kategori | Train | Test |
|---:|---|---:|---:|
| 0 | T-shirt/top | 6.000 | 1.000 |
| 1 | Trouser | 6.000 | 1.000 |
| 2 | Pullover | 6.000 | 1.000 |
| 3 | Dress | 6.000 | 1.000 |
| 4 | Coat | 6.000 | 1.000 |
| 5 | Sandal | 6.000 | 1.000 |
| 6 | Shirt | 6.000 | 1.000 |
| 7 | Sneaker | 6.000 | 1.000 |
| 8 | Bag | 6.000 | 1.000 |
| 9 | Ankle boot | 6.000 | 1.000 |

### 4.3 Statistik Nilai Piksel

| Statistik | Train | Test |
|---|---:|---:|
| Minimum | 0 | 0 |
| Maksimum | 255 | 255 |
| Rata-rata | 72,956831 | 73,160805 |

Nilai piksel berada pada rentang 0–255.

### 4.4 Visualisasi Gambar

Beberapa sampel gambar berhasil direkonstruksi dari 784 fitur piksel menjadi gambar grayscale berukuran 28 × 28 piksel.

Visualisasi digunakan untuk memeriksa bahwa data numerik dapat direpresentasikan sebagai gambar dan label dapat ditampilkan bersama kategori yang sesuai.

## 5. Audit Duplikasi Gambar

### 5.1 Metode Pemeriksaan

Audit dilakukan berdasarkan 784 fitur piksel tanpa menyertakan kolom label dalam identifikasi gambar.

Hash digunakan untuk menemukan kandidat gambar yang kemungkinan identik. Setiap kandidat kemudian diverifikasi dengan membandingkan seluruh nilai piksel secara langsung.

Dataset asli tidak diubah selama proses audit.

### 5.2 Hasil Audit

| Pemeriksaan | Hasil |
|---|---:|
| Kelompok gambar identik dalam train | 43 |
| Kelompok gambar identik dalam test | 1 |
| Kelompok gambar identik lintas train-test | 10 |
| Konflik label dalam train | 0 |
| Konflik label dalam test | 0 |
| Konflik label lintas train-test | 1 |

Jumlah tersebut merupakan jumlah kelompok gambar identik berdasarkan hasil audit, bukan jumlah seluruh baris yang terlibat.

### 5.3 Verifikasi Inkonsistensi Label

Ditemukan satu kelompok gambar identik lintas dataset dengan label berbeda.

| Keterangan | Train | Test |
|---|---|---|
| Indeks baris | 52205 | 9560 |
| Label | 6 (Shirt) | 0 (T-shirt/top) |
| Jumlah fitur piksel | 784 | 784 |

Verifikasi langsung terhadap seluruh 784 nilai piksel menghasilkan:

- Hasil perbandingan: `True`
- Jumlah piksel berbeda: `0`

Dengan demikian, kedua baris tersebut memiliki gambar yang benar-benar identik, tetapi label yang berbeda.

### 5.4 Interpretasi Temuan

Temuan tersebut menunjukkan adanya inkonsistensi label pada gambar identik.

Selain itu, ditemukan 10 kelompok gambar identik yang muncul pada data train dan test. Kondisi ini merupakan potensi data leakage karena gambar yang terdapat dalam data test juga muncul dalam data train.

Dampak temuan tersebut terhadap performa dan validitas evaluasi model belum diukur karena tahap pelatihan model belum dilakukan.

## 6. Keputusan dan Catatan

Pada tahap Dataset Exploration:

1. Dataset asli tetap dipertahankan.
2. Tidak ada gambar yang dihapus.
3. Tidak ada label yang diubah.
4. Tidak dilakukan pemindahan sampel antara train dan test.
5. Temuan duplikasi dan inkonsistensi label dicatat untuk dibahas bersama kelompok.

Kebijakan penanganan duplikasi internal, duplikasi lintas dataset, dan inkonsistensi label belum ditetapkan. Keputusan tersebut perlu disepakati sebelum tahap preprocessing.

## 7. Status Eksperimen

| Aktivitas | Status |
|---|---|
| Pembacaan dataset | Selesai |
| Pemeriksaan struktur dan dimensi | Selesai |
| Pemeriksaan tipe data | Selesai |
| Pemeriksaan missing values | Selesai |
| Pemeriksaan duplikasi baris | Selesai |
| Audit duplikasi berdasarkan 784 piksel | Selesai |
| Audit gambar identik lintas dataset | Selesai |
| Verifikasi konflik label | Selesai |
| Pemeriksaan distribusi kelas | Selesai |
| Statistik nilai piksel | Selesai |
| Rekonstruksi dan visualisasi gambar | Selesai |
| Validasi struktur kolom | Selesai |
| Penyusunan kesimpulan notebook | Selesai |
| Persetujuan kebijakan penanganan data | Menunggu diskusi kelompok |
| Tahap preprocessing | Belum dimulai |

## 8. Rencana Selanjutnya

1. Meninjau hasil Dataset Exploration bersama seluruh anggota kelompok.
2. Menentukan kebijakan penanganan duplikasi dan inkonsistensi label.
3. Mendokumentasikan keputusan kelompok.
4. Melanjutkan ke preprocessing setelah mendapat persetujuan kelompok.
5. Memastikan seluruh perubahan dokumentasi dan notebook ditinjau sebelum commit ke GitHub.

---

**Catatan:** Experiment log ini mencatat hasil aktual Dataset Exploration. Setiap perubahan kebijakan dan hasil eksperimen berikutnya akan ditambahkan sebagai catatan baru tanpa menghilangkan riwayat eksperimen sebelumnya.