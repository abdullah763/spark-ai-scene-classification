from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import CFG
from .data import build_dataloaders
from .models import build_model
from .utils import get_device, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the untouched test split.")
    parser.add_argument("--data-root", default=str(CFG.data_root))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--num-workers", type=int, default=CFG.num_workers)
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint["class_names"]
    model_name = checkpoint["model_name"]

    data = build_dataloaders(
        args.data_root,
        image_size=checkpoint.get("image_size", CFG.image_size),
        batch_size=args.batch_size,
        val_fraction=CFG.val_fraction,
        seed=CFG.seed,
        num_workers=args.num_workers,
    )
    if class_names != data.class_names:
        raise ValueError(f"Checkpoint classes {class_names} != dataset classes {data.class_names}")

    model = build_model(model_name, len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    y_true, y_pred, confidences, margins = [], [], [], []
    with torch.inference_mode():
        for images, targets in data.test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            top2 = probs.topk(k=2, dim=1)
            preds = top2.indices[:, 0]
            y_true.extend(targets.tolist())
            y_pred.extend(preds.cpu().tolist())
            confidences.extend(top2.values[:, 0].cpu().tolist())
            margins.extend((top2.values[:, 0] - top2.values[:, 1]).cpu().tolist())

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    metrics = {
        "model_name": model_name,
        "checkpoint": str(args.checkpoint),
        "test_images": len(y_true),
        "accuracy": accuracy_score(y_true_np, y_pred_np),
        "macro_precision": precision_score(y_true_np, y_pred_np, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true_np, y_pred_np, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true_np, y_pred_np, average="macro", zero_division=0),
    }

    out_metrics = CFG.outputs_dir / "metrics"
    out_plots = CFG.outputs_dir / "plots"
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true_np,
        y_pred_np,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(out_metrics / f"{model_name}_classification_report.csv")

    cm = confusion_matrix(y_true_np, y_pred_np)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix — {model_name}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_plots / f"{model_name}_confusion_matrix.png", dpi=170)
    plt.close(fig)

    # ImageFolder iteration order matches non-shuffled test_loader order.
    test_samples = data.test_loader.dataset.samples
    predictions = pd.DataFrame(
        {
            "path": [str(path) for path, _ in test_samples],
            "true_idx": y_true,
            "pred_idx": y_pred,
            "true_class": [class_names[i] for i in y_true],
            "pred_class": [class_names[i] for i in y_pred],
            "confidence": confidences,
            "margin": margins,
        }
    )
    predictions["correct"] = predictions["true_idx"] == predictions["pred_idx"]
    predictions.to_csv(out_metrics / f"{model_name}_test_predictions.csv", index=False)

    with (out_metrics / f"{model_name}_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    LOGGER.info("evaluation artifacts saved under %s", CFG.outputs_dir)


if __name__ == "__main__":
    main()
