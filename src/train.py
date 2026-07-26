"""
Core training loop implementing the staged/progressive unfreezing schedule
from Section 3(3): train the head first, then progressively unfreeze
deeper layers at a lower learning rate.

Used directly for a one-off run, and imported by tune.py / ablation.py so
the same training logic backs the hyperparameter search and the
augmentation ablation.
"""
import argparse
import copy
import time

import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from dataset import build_dataloader
from model import build_model
from transforms import eval_transform, get_train_transform
from utils import get_device, save_checkpoint, set_seed

# Staged schedule: (unfreeze_stage, epochs, lr_multiplier)
# lr_multiplier scales down the base LR as we open up more of the backbone,
# so already-good ImageNet features aren't clobbered by large gradient steps.
DEFAULT_SCHEDULE = [
    (0, 5, 1.0),   # head only
    (1, 5, 0.5),   # + layer4
    (2, 5, 0.25),  # + layer3
    (3, 5, 0.1),   # full fine-tune
]


def build_optimizer(name: str, params, lr: float):
    if name == "adam":
        return Adam(params, lr=lr)
    if name == "sgd_momentum":
        return SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


@torch.no_grad()
def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return loss_sum / total, correct / total


def train_model(
    train_csv: str,
    val_csv: str,
    lr: float = 1e-4,
    optimizer_name: str = "adam",
    batch_size: int = 32,
    dropout: float = 0.3,
    augment: bool = True,
    schedule=None,
    patience: int = 5,
    seed: int = 42,
    checkpoint_path: str | None = None,
    verbose: bool = True,
):
    """Runs the full staged fine-tuning schedule. Returns the best model
    (by validation accuracy, ties by lower validation loss) and a history
    dict, so callers (Optuna, ablation) can inspect val performance without
    ever touching the test split."""
    set_seed(seed)
    device = get_device()
    schedule = schedule or DEFAULT_SCHEDULE

    train_loader = build_dataloader(
        train_csv, get_train_transform(augment), batch_size, shuffle=True
    )
    val_loader = build_dataloader(val_csv, eval_transform(), batch_size, shuffle=False)

    model = build_model(dropout=dropout, device=device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_val_loss = -1.0, float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for stage, n_epochs, lr_mult in schedule:
        model.set_unfreeze_stage(stage)
        optimizer = build_optimizer(
            optimizer_name, model.trainable_parameters(), lr * lr_mult
        )

        for epoch in range(n_epochs):
            model.train()
            running_loss, running_correct, seen = 0.0, 0, 0
            t0 = time.time()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_correct += (logits.argmax(dim=1) == labels).sum().item()
                seen += images.size(0)

            train_loss = running_loss / seen
            train_acc = running_correct / seen
            val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)

            history.append({
                "stage": stage, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            })
            if verbose:
                print(
                    f"[stage {stage}] epoch {epoch+1}/{n_epochs} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                    f"({time.time()-t0:.1f}s)"
                )

            # Early stopping tracks val_acc first, val_loss as tiebreaker/overfit check.
            improved = val_acc > best_val_acc or (
                val_acc == best_val_acc and val_loss < best_val_loss
            )
            if improved:
                best_val_acc, best_val_loss = val_acc, val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping (no improvement for {patience} epochs).")
                    break

    model.load_state_dict(best_state)
    if checkpoint_path:
        save_checkpoint(model, checkpoint_path, extra={
            "val_acc": best_val_acc, "val_loss": best_val_loss,
            "hparams": {
                "lr": lr, "optimizer": optimizer_name, "batch_size": batch_size,
                "dropout": dropout, "augment": augment,
            },
        })
    return model, {"best_val_acc": best_val_acc, "best_val_loss": best_val_loss, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd_momentum"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = ap.parse_args()

    _, result = train_model(
        train_csv=f"{args.splits_dir}/train.csv",
        val_csv=f"{args.splits_dir}/val.csv",
        lr=args.lr,
        optimizer_name=args.optimizer,
        batch_size=args.batch_size,
        dropout=args.dropout,
        augment=not args.no_augment,
        checkpoint_path=args.checkpoint,
    )
    print(f"\nBest val_acc={result['best_val_acc']:.4f}, val_loss={result['best_val_loss']:.4f}")
    print(f"Checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
