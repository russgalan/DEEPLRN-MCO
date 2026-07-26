# Image-Based Waste Classification with ResNet50 (TrashNet)

Code implementation of the methodology described in "Image-Based Waste
Classification with Convolutional Neural Networks" (Borromeo, Galan, Santos).

## 1. Dataset setup

Download TrashNet from Kaggle:
https://www.kaggle.com/datasets/feyzazkefe/trashnet
(this mirror already unzips the original garythung/trashnet repo into
per-class folders, which is what this code expects).

Expected folder layout after download:

```
data/raw/
    cardboard/
    glass/
    metal/
    paper/
    plastic/
    trash/
```

If you're on Colab, the fastest path is the Kaggle API:

```bash
pip install kaggle
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/   # your Kaggle API token
kaggle datasets download -d feyzazkefe/trashnet -p data/ --unzip
```

You should end up with 2,527 images across 6 classes, with a known
class imbalance (glass/paper/plastic are larger classes, trash is the
smallest — keep this in mind, it's why the paper insists on macro
precision/recall/F1 and a stratified split).

## 2. Split the data (70/15/15, stratified)

```bash
python src/make_splits.py --raw_dir data/raw --out_dir data/splits
```

This writes `train.csv`, `val.csv`, `test.csv` (filepath,label) using a
stratified split so class proportions are preserved in all three sets,
and the test split is written once and never touched again until final
reporting — exactly as described in Section 4.1.

## 3. Hyperparameter search (Section 4.2)

```bash
python src/tune.py --splits_dir data/splits --n_trials 30
```

Runs Optuna over: learning rate {1e-3, 1e-4, 1e-5}, optimizer {adam, sgd_momentum},
batch size {16, 32, 64}, dropout {0.2, 0.3, 0.5}. Selects the trial with
best validation accuracy (ties broken by lower validation loss). Saves
`best_params.json`.

## 4. Augmentation ablation (Section 4.3)

```bash
python src/ablation.py --splits_dir data/splits --best_params outputs/best_params.json
```

Trains two models with the winning hyperparameters — one with heavy
augmentation, one with none — and evaluates both on the held-out test
split.

## 5. Final evaluation (Section 4.4)

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt --splits_dir data/splits
```

Reports overall accuracy, per-class + macro-averaged precision/recall/F1,
and a confusion matrix (saved as a PNG).

## Files

- `src/dataset.py` — stratified split loader + `TrashNetDataset`
- `src/transforms.py` — preprocessing + heavy/none augmentation pipelines
- `src/model.py` — ResNet50 backbone, staged/progressive unfreezing
- `src/train.py` — training/validation loop, early stopping, checkpointing
- `src/tune.py` — Optuna hyperparameter search
- `src/ablation.py` — augmentation ablation study
- `src/evaluate.py` — final metrics + confusion matrix
- `src/make_splits.py` — builds the stratified 70/15/15 CSV splits
- `src/utils.py` — seeding, misc helpers
