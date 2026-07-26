"""
Section 4.3 - Data Augmentation Ablation.

Using the best hyperparameter configuration from tune.py, trains two
otherwise-identical models -- one with heavy augmentation (rotation, zoom,
flip, brightness), one with none -- and evaluates BOTH on the same
held-out test split, to see whether augmentation actually helps
generalization on this small dataset.
"""
import argparse

from evaluate import full_evaluate
from train import train_model
from utils import load_json, save_json


def run_ablation(splits_dir: str, best_params: dict, out_dir: str = "outputs"):
    train_csv = f"{splits_dir}/train.csv"
    val_csv = f"{splits_dir}/val.csv"
    test_csv = f"{splits_dir}/test.csv"

    results = {}
    for augment in (True, False):
        arm = "heavy_augmentation" if augment else "no_augmentation"
        print(f"\n=== Training arm: {arm} ===")
        model, train_result = train_model(
            train_csv=train_csv,
            val_csv=val_csv,
            lr=best_params["lr"],
            optimizer_name=best_params["optimizer"],
            batch_size=best_params["batch_size"],
            dropout=best_params["dropout"],
            augment=augment,
            checkpoint_path=f"checkpoints/ablation_{arm}.pt",
        )
        print(f"[{arm}] best val_acc={train_result['best_val_acc']:.4f}")

        test_metrics = full_evaluate(
            model, test_csv, batch_size=best_params["batch_size"],
            save_confusion_matrix_path=f"{out_dir}/confusion_matrix_{arm}.png",
        )
        results[arm] = {
            "val_acc": train_result["best_val_acc"],
            "val_loss": train_result["best_val_loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_macro_f1": test_metrics["macro_f1"],
            "per_class": test_metrics["per_class"],
        }
        print(f"[{arm}] test_accuracy={test_metrics['accuracy']:.4f} "
              f"macro_f1={test_metrics['macro_f1']:.4f}")

    save_json(results, f"{out_dir}/ablation_results.json")
    print(f"\nAblation results saved to {out_dir}/ablation_results.json")
    print(f"Confusion matrices saved to {out_dir}/confusion_matrix_<arm>.png")

    diff = (results["heavy_augmentation"]["test_accuracy"]
            - results["no_augmentation"]["test_accuracy"])
    print(f"\nAugmentation effect on test accuracy: {diff:+.4f} "
          f"({'helped' if diff > 0 else 'hurt' if diff < 0 else 'no change'})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--best_params", default="outputs/best_params.json",
                     help="Path to best_params.json produced by tune.py")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    params_blob = load_json(args.best_params)
    best_params = params_blob["best_params"]

    run_ablation(args.splits_dir, best_params, args.out_dir)


if __name__ == "__main__":
    main()
