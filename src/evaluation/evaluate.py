
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.data.dataset import (
    TEST_CSV,
    CLASS_NAMES,
    FashionMNISTDataset,
    load_csv,
    preprocess_data,
)

from src.models.vit import VisionTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_PATH = OUTPUT_DIR / "best_vit.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"

BATCH_SIZE = 64
DEVICE = torch.device("cpu")

NUM_CLASSES = 10


# ============================================================
# LOAD TEST DATA
# ============================================================

def create_test_loader():
    """
    Memuat seluruh test set untuk evaluasi akhir.
    Tidak melakukan tuning atau perubahan data.
    """

    images, labels, _ = load_csv(TEST_CSV)

    images, labels = preprocess_data(images, labels)

    test_dataset = FashionMNISTDataset(images, labels)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return test_loader


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

def load_best_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}"
        )

    model = VisionTransformer().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("[CHECKPOINT INFORMATION]")
    print(f"Checkpoint epoch : {checkpoint['epoch']}")
    print(
        "Best validation accuracy : "
        f"{checkpoint['validation_accuracy'] * 100:.2f}%"
    )

    return model


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_model(model, test_loader):
    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

    average_loss = total_loss / total_samples

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    metrics = {
        "test_samples": total_samples,
        "test_loss": average_loss,
        "test_accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1_score": f1,
    }

    return (
        metrics,
        cm,
        report,
        report_text,
        all_labels,
        all_predictions,
    )


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

def save_evaluation_results(
    metrics,
    cm,
    report,
    report_text,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save main metrics
    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Save classification report as TXT
    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    # Save classification report as CSV
    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv",
        index=True,
    )

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    cm_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    print("\n[EVALUATION RESULTS]")
    print(f"Test samples : {metrics['test_samples']}")
    print(f"Test loss    : {metrics['test_loss']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Precision    : {metrics['macro_precision']:.4f}")
    print(f"Recall       : {metrics['macro_recall']:.4f}")
    print(f"F1-score     : {metrics['macro_f1_score']:.4f}")

    print("\n[CLASSIFICATION REPORT]")
    print(report_text)


# ============================================================
# TRAINING LOSS VISUALIZATION
# ============================================================

def plot_training_loss():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Training history tidak ditemukan: {HISTORY_PATH}"
        )

    history = pd.read_csv(HISTORY_PATH)

    required_columns = [
        "epoch",
        "train_loss",
        "validation_loss",
    ]

    for column in required_columns:
        if column not in history.columns:
            raise ValueError(
                f"Kolom {column} tidak ditemukan di training history."
            )

    plt.figure(figsize=(9, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        history["epoch"],
        history["validation_loss"],
        marker="o",
        label="Validation Loss",
    )

    plt.title("ViT Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.xticks(history["epoch"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_loss.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# TRAINING ACCURACY VISUALIZATION
# ============================================================

def plot_training_accuracy():
    history = pd.read_csv(HISTORY_PATH)

    required_columns = [
        "epoch",
        "train_accuracy",
        "validation_accuracy",
    ]

    for column in required_columns:
        if column not in history.columns:
            raise ValueError(
                f"Kolom {column} tidak ditemukan di training history."
            )

    plt.figure(figsize=(9, 5))

    plt.plot(
        history["epoch"],
        history["train_accuracy"] * 100,
        marker="o",
        label="Training Accuracy",
    )

    plt.plot(
        history["epoch"],
        history["validation_accuracy"] * 100,
        marker="o",
        label="Validation Accuracy",
    )

    plt.title("ViT Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.xticks(history["epoch"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_accuracy.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

def plot_confusion_matrix(cm):
    plt.figure(figsize=(11, 9))

    plt.imshow(cm, interpolation="nearest")
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.colorbar(label="Number of Predictions")

    positions = np.arange(NUM_CLASSES)

    plt.xticks(
        positions,
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        positions,
        CLASS_NAMES,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = cm.max() / 2.0

    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            plt.text(
                col,
                row,
                format(cm[row, col], "d"),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[row, col] > threshold
                    else "black"
                ),
                fontsize=8,
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("VISION TRANSFORMER - FINAL TEST EVALUATION")
    print("=" * 65)

    print(f"Device: {DEVICE}")

    # Load checkpoint
    model = load_best_model()

    # Load test data only for final evaluation
    test_loader = create_test_loader()

    print(f"\nTest samples: {len(test_loader.dataset)}")

    # Evaluate
    (
        metrics,
        cm,
        report,
        report_text,
        all_labels,
        all_predictions,
    ) = evaluate_model(model, test_loader)

    # Save results
    save_evaluation_results(
        metrics,
        cm,
        report,
        report_text,
    )

    # Generate plots
    plot_training_loss()
    plot_training_accuracy()
    plot_confusion_matrix(cm)

    print("\n[VISUALIZATION]")
    print("Training loss      : outputs/training_loss.png")
    print("Training accuracy  : outputs/training_accuracy.png")
    print("Confusion matrix   : outputs/confusion_matrix.png")

    print("\n[EVALUATION FILES]")
    print("Metrics            : outputs/evaluation_metrics.json")
    print("Classification TXT : outputs/classification_report.txt")
    print("Classification CSV : outputs/classification_report.csv")
    print("Confusion CSV      : outputs/confusion_matrix.csv")

    print("\n" + "=" * 65)
    print("EVALUATION AND VISUALIZATION COMPLETED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
