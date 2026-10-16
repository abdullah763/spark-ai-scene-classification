from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from .transforms import build_eval_transform, build_train_transform


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    train_size: int
    val_size: int
    test_size: int
    train_dir: Path
    test_dir: Path


def _resolve_split_dir(data_root: str | Path, split: str) -> Path:
    root = Path(data_root)
    candidates = [
        root / split / split,
        root / split,
        root,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(p.is_dir() for p in candidate.iterdir()):
            # The root itself is accepted only if it visibly contains class folders.
            if candidate == root and split not in root.name:
                continue
            return candidate
    raise FileNotFoundError(
        f"Could not locate '{split}' under {root}. "
        f"Expected e.g. {root / split / split} or {root / split}."
    )


def resolve_dataset_dirs(data_root: str | Path) -> tuple[Path, Path]:
    root = Path(data_root)
    train_candidates = [root / "seg_train" / "seg_train", root / "seg_train"]
    test_candidates = [root / "seg_test" / "seg_test", root / "seg_test"]

    train_dir = next((p for p in train_candidates if p.is_dir()), None)
    test_dir = next((p for p in test_candidates if p.is_dir()), None)
    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            "Intel dataset folders were not found. Expected seg_train/seg_train and "
            "seg_test/seg_test under the configured data root."
        )
    return train_dir, test_dir


def build_dataloaders(
    data_root: str | Path,
    image_size: int = 224,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
) -> DataBundle:
    train_dir, test_dir = resolve_dataset_dirs(data_root)

    split_source = ImageFolder(train_dir)
    targets = np.asarray(split_source.targets)
    indices = np.arange(len(split_source))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=targets,
    )

    train_full = ImageFolder(train_dir, transform=build_train_transform(image_size))
    val_full = ImageFolder(train_dir, transform=build_eval_transform(image_size))
    test_ds = ImageFolder(test_dir, transform=build_eval_transform(image_size))

    if train_full.classes != test_ds.classes:
        raise ValueError(
            f"Class mismatch between train and test folders: "
            f"{train_full.classes} vs {test_ds.classes}"
        )

    train_ds = Subset(train_full, train_idx.tolist())
    val_ds = Subset(val_full, val_idx.tolist())

    pin_memory = True
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=train_full.classes,
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
        train_dir=train_dir,
        test_dir=test_dir,
    )
