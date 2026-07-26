"""
Section 4.2 - Hyperparameter Tuning.

Uses Optuna to search:
  - Learning Rate:  {1e-3, 1e-4, 1e-5}
  - Optimizer:      {adam, sgd_momentum}
  - Batch Size:     {16, 32, 64}
  - Dropout Rate:   {0.2, 0.3, 0.5}

Each trial is scored on validation accuracy (val loss used as a tiebreaker
and to flag overfitting -- see train.py's early-stopping logic, which
already tracks both). The winning config is written to best_params.json;
the held-out test split is never touched here.
"""
import argparse

import optuna

from train import train_model
from utils import save_json


def make_objective(splits_dir: str, augment: bool, patience: int):
    train_csv = f"{splits_dir}/train.csv"
    val_csv = f"{splits_dir}/val.csv"

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_categorical("lr", [1e-3, 1e-4, 1e-5])
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd_momentum"])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout = trial.suggest_categorical("dropout", [0.2, 0.3, 0.5])

        _, result = train_model(
            train_csv=train_csv,
            val_csv=val_csv,
            lr=lr,
            optimizer_name=optimizer_name,
            batch_size=batch_size,
            dropout=dropout,
            augment=augment,
            patience=patience,
            verbose=False,
        )

        # Stash val_loss on the trial so we can use it as a tiebreaker /
        # overfitting check when comparing trials with tied accuracy.
        trial.set_user_attr("val_loss", result["best_val_loss"])
        return result["best_val_acc"]

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3,
                     help="Shorter patience during search to keep trials cheap.")
    ap.add_argument("--augment", action="store_true", default=True,
                     help="Use heavy augmentation during the search "
                          "(the augmentation ablation itself happens separately).")
    ap.add_argument("--out", default="outputs/best_params.json")
    args = ap.parse_args()

    study = optuna.create_study(direction="maximize", study_name="waste_resnet50_tuning")
    study.optimize(
        make_objective(args.splits_dir, args.augment, args.patience),
        n_trials=args.n_trials,
    )

    print("Best trial:")
    print(f"  val_acc:  {study.best_value:.4f}")
    print(f"  val_loss: {study.best_trial.user_attrs.get('val_loss'):.4f}")
    print(f"  params:   {study.best_params}")

    save_json({
        "best_params": study.best_params,
        "best_val_acc": study.best_value,
        "best_val_loss": study.best_trial.user_attrs.get("val_loss"),
    }, args.out)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
