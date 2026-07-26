"""
Section 4.4 - Evaluation Metrics.

Reports, on the held-out test split only:
  - Overall accuracy
  - Per-class and macro-averaged precision / recall / F1
  - Confusion matrix (saved as PNG), so misclassification patterns
    like "glass mistaken for clear plastic" are visible directly.
"""
import argparse

import matplotlib

matplotlib.use("Agg")  # headless-safe (Colab/servers)
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from dataset import build_dataloader
from model import build_model
from transforms import eval_transform
from utils import CLASS_NAMES, get_device, load_checkpoint


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (Test Split)")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def full_evaluate(
    model,
    test_csv: str,
    batch_size: int = 32,
    save_confusion_matrix_path: str = "outputs/confusion_matrix.png",
) -> dict:
    device = get_device()
    model = model.to(device)
    test_loader = build_dataloader(test_csv, eval_transform(), batch_size, shuffle=False)

    y_true, y_pred = collect_predictions(model, test_loader, device)

    accuracy = float((y_true == y_pred).mean())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    per_class = {
        CLASS_NAMES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(CLASS_NAMES))
    }

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    plot_confusion_matrix(cm, CLASS_NAMES, save_confusion_matrix_path)

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3,
                     help="Must match the dropout the checkpoint was trained with.")
    ap.add_argument("--out", default="outputs/test_metrics.png")
    args = ap.parse_args()

    device = get_device()
    model = build_model(dropout=args.dropout, device=device)
    payload = load_checkpoint(model, args.checkpoint, map_location=device)
    if "hparams" in payload:
        print(f"Loaded checkpoint trained with: {payload['hparams']}")

    metrics = full_evaluate(
        model, f"{args.splits_dir}/test.csv",
        batch_size=args.batch_size, save_confusion_matrix_path=args.out,
    )

    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}\n")
    print("Per-class breakdown:")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:10s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} support={m['support']}")
    print(f"\nConfusion matrix saved to {args.out}")


if __name__ == "__main__":
    main()
