from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .config import CFG
from .data import resolve_dataset_dirs
from .utils import setup_logging

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def audit_split(split_dir: Path):
    rows = []
    corrupt = []
    size_counts = Counter()
    mode_counts = Counter()

    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    size_counts[str(image.size)] += 1
                    mode_counts[image.mode] += 1
                    image.verify()
                rows.append({"class": class_dir.name, "path": str(path), "valid": True})
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                corrupt.append({"path": str(path), "error": str(exc)})
                rows.append({"class": class_dir.name, "path": str(path), "valid": False})

    return pd.DataFrame(rows), corrupt, size_counts, mode_counts


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Audit Intel scene-classification dataset.")
    parser.add_argument("--data-root", default=str(CFG.data_root))
    args = parser.parse_args()

    train_dir, test_dir = resolve_dataset_dirs(args.data_root)
    out_metrics = CFG.outputs_dir / "metrics"
    out_plots = CFG.outputs_dir / "plots"
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    summary = {"train_dir": str(train_dir), "test_dir": str(test_dir), "splits": {}}
    all_counts = []

    for name, split_dir in [("train_source", train_dir), ("test", test_dir)]:
        LOGGER.info("auditing %s: %s", name, split_dir)
        frame, corrupt, sizes, modes = audit_split(split_dir)
        counts = frame[frame["valid"]].groupby("class").size().sort_index()
        counts.to_csv(out_metrics / f"{name}_class_counts.csv", header=["count"])
        all_counts.append(pd.DataFrame({"class": counts.index, "count": counts.values, "split": name}))
        summary["splits"][name] = {
            "images": int(frame["valid"].sum()),
            "corrupt_images": len(corrupt),
            "class_counts": {k: int(v) for k, v in counts.to_dict().items()},
            "top_image_sizes": dict(sizes.most_common(10)),
            "image_modes": dict(modes),
            "corrupt_examples": corrupt[:20],
        }

    combined = pd.concat(all_counts, ignore_index=True)
    pivot = combined.pivot(index="class", columns="split", values="count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=30)
    ax.set_title("Intel Scene Dataset Class Distribution")
    ax.set_ylabel("Images")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_plots / "class_distribution.png", dpi=160)
    plt.close(fig)

    with (out_metrics / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("dataset audit saved to %s", out_metrics / "dataset_audit.json")


if __name__ == "__main__":
    main()
