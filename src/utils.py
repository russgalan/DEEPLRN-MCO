"""Shared helpers: reproducibility, device selection, checkpoint I/O."""
import json
import random
from pathlib import Path

import numpy as np
import torch

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def set_seed(seed: int = 42) -> None:
    """Fix all relevant RNGs so runs (splits, init, augmentation) are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: torch.nn.Module, path: str, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location=None) -> dict:
    payload = torch.load(path, map_location=map_location or get_device())
    model.load_state_dict(payload["model_state"])
    return payload


def save_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
