
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "fashion-mnist_train.csv"
TEST_CSV = PROJECT_ROOT / "data" / "raw" / "fashion-mnist_test.csv"

BATCH_SIZE = 64
NUM_WORKERS = 0  # Aman untuk Windows dan CPU

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# ============================================================
# DATASET CLASS
# ============================================================

class FashionMNISTDataset(Dataset):
    """
    PyTorch Dataset untuk Fashion-MNIST.

    Setiap gambar memiliki bentuk [1, 28, 28]
    dengan nilai piksel yang sudah dinormalisasi ke 0-1.
    """

    def __init__(self, images, labels):
        if not isinstance(images, torch.Tensor):
            raise TypeError("Images harus berupa torch.Tensor.")

        if not isinstance(labels, torch.Tensor):
            raise TypeError("Labels harus berupa torch.Tensor.")

        if images.ndim != 4:
            raise ValueError(
                f"Dimensi images harus 4D, ditemukan {images.shape}."
            )

        if images.shape[1:] != (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Bentuk gambar tidak sesuai: {images.shape}."
            )

        if labels.ndim != 1:
            raise ValueError("Labels harus memiliki satu dimensi.")

        if len(images) != len(labels):
            raise ValueError("Jumlah images dan labels tidak sama.")

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


# ============================================================
# CSV LOADING AND PREPROCESSING
# ============================================================

def load_csv(csv_path):
    """
    Membaca CSV dan memisahkan label dengan fitur piksel.
    Dataset mentah hanya dibaca, tidak dimodifikasi.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"File dataset tidak ditemukan: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError(
            f"Kolom 'label' tidak ditemukan pada {csv_path.name}."
        )

    pixel_columns = [col for col in df.columns if col != "label"]

    if len(pixel_columns) != NUM_PIXELS:
        raise ValueError(
            f"Jumlah fitur piksel harus {NUM_PIXELS}, "
            f"ditemukan {len(pixel_columns)}."
        )

    if df.isnull().values.any():
        raise ValueError(
            f"Ditemukan missing value pada {csv_path.name}."
        )

    images = df[pixel_columns].to_numpy(dtype="float32", copy=True)
    labels = df["label"].to_numpy(dtype="int64", copy=True)

    if images.size > 0:
        if images.min() < 0 or images.max() > 255:
            raise ValueError(
                "Nilai piksel harus berada pada rentang 0-255."
            )

    if labels.size > 0:
        if labels.min() < 0 or labels.max() >= len(CLASS_NAMES):
            raise ValueError(
                "Label harus berada pada rentang 0-9."
            )

    return images, labels, pixel_columns



def preprocess_data(images, labels):
    """
    Normalisasi piksel dan mengubah gambar menjadi tensor.

    Input images : [N, 784]
    Output images: [N, 1, 28, 28]
    """

    image_tensor = torch.from_numpy(images)

    # Normalisasi nilai piksel dari 0-255 menjadi 0-1
    image_tensor = image_tensor / 255.0

    # Mengubah bentuk menjadi [N, 1, 28, 28]
    image_tensor = image_tensor.reshape(
        -1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE
    )

    # Membuat salinan tensor label yang writable
    label_tensor = torch.tensor(labels, dtype=torch.int64)

    return image_tensor, label_tensor



# ============================================================
# DATALOADER CREATION
# ============================================================

def create_dataloaders(batch_size=BATCH_SIZE):
    """
    Membuat Dataset dan DataLoader untuk train dan test.
    """

    train_images, train_labels, train_columns = load_csv(TRAIN_CSV)
    test_images, test_labels, test_columns = load_csv(TEST_CSV)

    # Memastikan urutan fitur train dan test konsisten
    if train_columns != test_columns:
        raise ValueError(
            "Urutan atau nama kolom fitur train dan test berbeda."
        )

    train_images, train_labels = preprocess_data(
        train_images, train_labels
    )

    test_images, test_labels = preprocess_data(
        test_images, test_labels
    )

    train_dataset = FashionMNISTDataset(
        train_images, train_labels
    )

    test_dataset = FashionMNISTDataset(
        test_images, test_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return train_loader, test_loader


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(train_loader, test_loader):
    """
    Memverifikasi dimensi, tipe data, rentang piksel,
    jumlah data, dan label.
    """

    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    print("=" * 60)
    print("FASHION-MNIST PREPROCESSING VALIDATION")
    print("=" * 60)

    print("\n[DATASET INFORMATION]")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")

    print("\n[TENSOR INFORMATION]")
    print(f"Train shape   : {train_dataset.images.shape}")
    print(f"Test shape    : {test_dataset.images.shape}")
    print(f"Image dtype   : {train_dataset.images.dtype}")
    print(f"Label dtype   : {train_dataset.labels.dtype}")

    print("\n[PIXEL RANGE]")
    print(
        f"Train range   : "
        f"{train_dataset.images.min().item():.4f} - "
        f"{train_dataset.images.max().item():.4f}"
    )
    print(
        f"Test range    : "
        f"{test_dataset.images.min().item():.4f} - "
        f"{test_dataset.images.max().item():.4f}"
    )

    print("\n[LABEL VALIDATION]")
    print(
        f"Train labels  : "
        f"{torch.unique(train_dataset.labels).tolist()}"
    )
    print(
        f"Test labels   : "
        f"{torch.unique(test_dataset.labels).tolist()}"
    )

    # Validasi tensor
    assert train_dataset.images.shape == (60000, 1, 28, 28)
    assert test_dataset.images.shape == (10000, 1, 28, 28)

    assert train_dataset.images.dtype == torch.float32
    assert test_dataset.images.dtype == torch.float32

    assert train_dataset.labels.dtype == torch.int64
    assert test_dataset.labels.dtype == torch.int64

    assert train_dataset.images.min() >= 0
    assert train_dataset.images.max() <= 1

    assert test_dataset.images.min() >= 0
    assert test_dataset.images.max() <= 1

    assert len(train_dataset) == 60000
    assert len(test_dataset) == 10000

    # Validasi satu batch DataLoader
    batch_images, batch_labels = next(iter(train_loader))

    print("\n[DATALOADER VALIDATION]")
    print(f"Batch image shape : {batch_images.shape}")
    print(f"Batch label shape : {batch_labels.shape}")
    print(f"Batch size        : {train_loader.batch_size}")

    assert batch_images.shape == (BATCH_SIZE, 1, 28, 28)
    assert batch_labels.shape == (BATCH_SIZE,)

    print("\n" + "=" * 60)
    print("PREPROCESSING VALIDATION PASSED!")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_loader, test_loader = create_dataloaders()

    validate_dataset(train_loader, test_loader)
