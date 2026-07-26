"""TrashNet Dataset + DataLoader construction from the CSV splits."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils import CLASS_NAMES

LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class TrashNetDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = LABEL_TO_IDX[row["label"]]
        return image, torch.tensor(label, dtype=torch.long)


def build_dataloader(
    csv_path: str,
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
) -> DataLoader:
    ds = TrashNetDataset(csv_path, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def class_counts(csv_path: str) -> pd.Series:
    """Useful for sanity-checking stratification and for optional class weighting."""
    df = pd.read_csv(csv_path)
    return df["label"].value_counts()
