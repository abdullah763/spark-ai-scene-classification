from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    seed: int = 42
    image_size: int = 224
    val_fraction: float = 0.15
    batch_size: int = 64
    num_workers: int = 2
    weight_decay: float = 1e-4
    num_classes: int = 6

    @property
    def data_root(self) -> Path:
        return Path(
            os.getenv(
                "SCENE_DATA_ROOT",
                "/kaggle/input/intel-image-classification",
            )
        )

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models"

    @property
    def outputs_dir(self) -> Path:
        return PROJECT_ROOT / "outputs"


CFG = Config()
