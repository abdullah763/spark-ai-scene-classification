from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import CFG
from .data import build_dataloaders
from .models import build_model
from .train import fit_model
from .utils import get_device, save_json, seed_everything, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Small, time-boxed MobileNetV3 LR tuning sweep.")
    parser.add_argument("--data-root", default=str(CFG.data_root))
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3, 3e-4, 1e-4])
    parser.add_argument("--epochs-per-trial", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--num-workers", type=int, default=CFG.num_workers)
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    device = get_device()
    data = build_dataloaders(
        args.data_root,
        image_size=CFG.image_size,
        batch_size=args.batch_size,
        val_fraction=CFG.val_fraction,
        seed=CFG.seed,
        num_workers=args.num_workers,
    )

    results = []
    tuning_dir = CFG.outputs_dir / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    for lr in args.learning_rates:
        seed_everything(CFG.seed)
        LOGGER.info("starting tuning trial lr=%g", lr)
        model = build_model(
            "mobilenet_v3_small",
            num_classes=len(data.class_names),
            pretrained=True,
            freeze_backbone=True,
        )
        ckpt = tuning_dir / f"mobilenet_frozen_lr_{lr:g}.pth"
        _, best_f1, elapsed = fit_model(
            model,
            data.train_loader,
            data.val_loader,
            data.class_names,
            model_name="mobilenet_v3_small",
            checkpoint_path=ckpt,
            epochs=args.epochs_per_trial,
            learning_rate=lr,
            weight_decay=CFG.weight_decay,
            device=device,
        )
        results.append(
            {
                "learning_rate": lr,
                "epochs": args.epochs_per_trial,
                "backbone": "frozen",
                "best_val_macro_f1": best_f1,
                "elapsed_seconds": elapsed,
                "checkpoint": str(ckpt),
            }
        )

    frame = pd.DataFrame(results).sort_values("best_val_macro_f1", ascending=False)
    frame.to_csv(tuning_dir / "tuning_results.csv", index=False)
    best = frame.iloc[0].to_dict()
    save_json(best, tuning_dir / "best_hparams.json")
    print(frame.to_string(index=False))
    print(f"\nBest frozen-backbone learning rate: {best['learning_rate']}")


if __name__ == "__main__":
    main()
