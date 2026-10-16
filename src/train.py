from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW

from .config import CFG
from .data import build_dataloaders
from .models import build_model
from .utils import get_device, save_json, seed_everything, setup_logging

LOGGER = logging.getLogger(__name__)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    running_loss = 0.0
    correct = 0
    total = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (preds == targets).sum().item()
        total += batch_size
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
    }


def fit_model(
    model,
    train_loader,
    val_loader,
    class_names,
    model_name: str,
    checkpoint_path: str | Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    history: list[dict] = []
    start = time.perf_counter()

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        LOGGER.info(
            "epoch=%d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch,
            epochs,
            row["train_loss"],
            row["train_accuracy"],
            row["val_loss"],
            row["val_accuracy"],
            row["val_macro_f1"],
        )

        if row["val_macro_f1"] > best_val_f1:
            best_val_f1 = row["val_macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "class_names": list(class_names),
                    "image_size": CFG.image_size,
                    "val_macro_f1": best_val_f1,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                },
                checkpoint_path,
            )

    elapsed = time.perf_counter() - start
    return history, best_val_f1, elapsed


def parse_args():
    parser = argparse.ArgumentParser(description="Train scene-classification models.")
    parser.add_argument("--data-root", default=str(CFG.data_root))
    parser.add_argument("--model", choices=["simple_cnn", "mobilenet_v3_small"], required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--num-workers", type=int, default=CFG.num_workers)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    seed_everything(CFG.seed)
    device = get_device()
    LOGGER.info("device=%s", device)

    data = build_dataloaders(
        args.data_root,
        image_size=CFG.image_size,
        batch_size=args.batch_size,
        val_fraction=CFG.val_fraction,
        seed=CFG.seed,
        num_workers=args.num_workers,
    )
    LOGGER.info(
        "classes=%s train=%d val=%d test=%d",
        data.class_names,
        data.train_size,
        data.val_size,
        data.test_size,
    )

    model = build_model(
        args.model,
        num_classes=len(data.class_names),
        pretrained=(not args.no_pretrained and args.model == "mobilenet_v3_small"),
        freeze_backbone=args.freeze_backbone,
    )

    checkpoint = args.checkpoint or str(CFG.models_dir / f"{args.model}_best.pth")
    history, best_f1, elapsed = fit_model(
        model,
        data.train_loader,
        data.val_loader,
        data.class_names,
        model_name=args.model,
        checkpoint_path=checkpoint,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    history_path = CFG.outputs_dir / "metrics" / f"{args.model}_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    summary = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "best_val_macro_f1": best_f1,
        "elapsed_seconds": elapsed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "freeze_backbone": args.freeze_backbone,
        "device": str(device),
    }
    save_json(summary, CFG.outputs_dir / "metrics" / f"{args.model}_training_summary.json")
    LOGGER.info("saved checkpoint=%s best_val_macro_f1=%.4f", checkpoint, best_f1)


if __name__ == "__main__":
    main()
