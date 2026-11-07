
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.dataset import (
    TRAIN_CSV,
    FashionMNISTDataset,
    load_csv,
    preprocess_data,
)
from src.models.vit import VisionTransformer


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

TRAIN_RATIO = 0.9
NUM_WORKERS = 0

DEVICE = torch.device("cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_PATH = OUTPUT_DIR / "best_vit.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA PREPARATION
# ============================================================

def create_train_validation_loaders():
    """
    Membaca hanya CSV train.
    Membagi 90% training dan 10% validation.
    Test set tidak dibaca atau digunakan.
    """

    images, labels, _ = load_csv(TRAIN_CSV)

    images, labels = preprocess_data(images, labels)

    full_train_dataset = FashionMNISTDataset(images, labels)

    train_size = int(TRAIN_RATIO * len(full_train_dataset))
    validation_size = len(full_train_dataset) - train_size

    split_generator = torch.Generator().manual_seed(SEED)

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=split_generator,
    )

    train_generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        generator=train_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return train_loader, validation_loader


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()

            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(model, optimizer, epoch, validation_accuracy):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_accuracy": validation_accuracy,
        "config": {
            "image_size": 28,
            "patch_size": 4,
            "num_patches": 49,
            "embed_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "mlp_dim": 128,
            "num_classes": 10,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
        },
    }

    torch.save(checkpoint, CHECKPOINT_PATH)


# ============================================================
# TRAINING HISTORY
# ============================================================

def save_history(history):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "epoch_duration_seconds",
    ]

    with open(
        HISTORY_PATH,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    set_seed()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("VISION TRANSFORMER - TRAINING")
    print("=" * 65)

    print(f"Device        : {DEVICE}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Learning rate : {LEARNING_RATE}")
    print(f"Random seed   : {SEED}")

    # Hanya memuat train CSV, bukan test CSV
    train_loader, validation_loader = (
        create_train_validation_loaders()
    )

    print("\n[DATA SPLIT]")
    print(f"Training samples   : {len(train_loader.dataset)}")
    print(f"Validation samples : {len(validation_loader.dataset)}")
    print("Test set           : Not loaded during training")

    model = VisionTransformer().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_accuracy = 0.0
    history = []

    print("\n[TRAINING START]")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
        )

        epoch_duration = time.perf_counter() - epoch_start

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "epoch_duration_seconds": epoch_duration,
        }

        history.append(epoch_result)

        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        print(f"Train Loss       : {train_loss:.4f}")
        print(f"Train Accuracy   : {train_accuracy * 100:.2f}%")
        print(f"Validation Loss  : {validation_loss:.4f}")
        print(f"Validation Acc   : {validation_accuracy * 100:.2f}%")
        print(f"Duration         : {epoch_duration:.2f} seconds")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            save_checkpoint(
                model,
                optimizer,
                epoch,
                validation_accuracy,
            )

            print("Checkpoint       : Best model saved")

        # Simpan history setelah setiap epoch agar progres tercatat
        save_history(history)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(f"Best checkpoint : {CHECKPOINT_PATH}")
    print(f"Training history: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
