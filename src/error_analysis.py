from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .config import CFG


def parse_args():
    parser = argparse.ArgumentParser(description="Quantitative and qualitative error analysis.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--top-n", type=int, default=24)
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    errors = frame[~frame["correct"].astype(bool)].copy()

    out_metrics = CFG.outputs_dir / "metrics"
    out_errors = CFG.outputs_dir / "error_analysis"
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_errors.mkdir(parents=True, exist_ok=True)

    by_true = (
        frame.groupby("true_class")["correct"]
        .agg(total="count", correct="sum")
        .reset_index()
    )
    by_true["errors"] = by_true["total"] - by_true["correct"]
    by_true["error_rate"] = by_true["errors"] / by_true["total"]
    by_true.to_csv(out_metrics / "errors_by_true_class.csv", index=False)

    confusion_pairs = (
        errors.groupby(["true_class", "pred_class"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confusion_pairs.to_csv(out_metrics / "confusion_pairs.csv", index=False)

    confidence_summary = {
        "total_predictions": int(len(frame)),
        "total_errors": int(len(errors)),
        "error_rate": float(len(errors) / max(len(frame), 1)),
        "mean_confidence_correct": float(frame.loc[frame["correct"].astype(bool), "confidence"].mean()),
        "mean_confidence_incorrect": float(errors["confidence"].mean()) if len(errors) else None,
        "high_confidence_errors_ge_0_90": int((errors["confidence"] >= 0.90).sum()),
    }
    with (out_metrics / "error_analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(confidence_summary, f, indent=2)

    selected = errors.sort_values("confidence", ascending=False).head(args.top_n)
    selected.to_csv(out_metrics / "highest_confidence_errors.csv", index=False)

    if len(selected):
        cols = 4
        rows = (len(selected) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
        axes = getattr(axes, "flat", [axes])
        for ax in axes:
            ax.axis("off")
        for ax, (_, row) in zip(axes, selected.iterrows()):
            try:
                image = Image.open(row["path"]).convert("RGB")
                ax.imshow(image)
                ax.set_title(
                    f"True: {row['true_class']}\nPred: {row['pred_class']} ({row['confidence']:.2f})",
                    fontsize=9,
                )
                ax.axis("off")
            except OSError:
                ax.set_title("Image unavailable")
        fig.suptitle("Highest-Confidence Misclassifications", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_errors / "highest_confidence_misclassifications.png", dpi=170)
        plt.close(fig)

    print(json.dumps(confidence_summary, indent=2))
    print("\nTop confusion pairs:")
    print(confusion_pairs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
