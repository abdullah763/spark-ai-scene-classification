from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import CFG
from .models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained PyTorch checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=str(CFG.models_dir / "best_model.onnx"))
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", CFG.image_size)
    model = build_model(checkpoint["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    classes_path = output.parent / "class_names.json"
    classes_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    # Optional numerical sanity check when onnxruntime is available.
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        ort_output = session.run(None, {"input": dummy.numpy().astype(np.float32)})[0]
        torch_output = model(dummy).detach().numpy()
        max_abs_diff = float(np.max(np.abs(ort_output - torch_output)))
        print(f"ONNX validation max_abs_diff={max_abs_diff:.6g}")
    except ImportError:
        print("onnxruntime not installed; skipped runtime comparison.")

    print(f"Exported: {output}")
    print(f"Classes:  {classes_path}")


if __name__ == "__main__":
    main()
