# Kaggle execution runbook

Use a GPU accelerator (T4 is sufficient). Add the public dataset `puneet6060/intel-image-classification` to the notebook.

## 0. Confirm paths

```python
from pathlib import Path
root = Path('/kaggle/input/intel-image-classification')
print(list(root.iterdir()))
print((root/'seg_train'/'seg_train').exists())
print((root/'seg_test'/'seg_test').exists())
```

## 1. Put project code in `/kaggle/working/spark-ai-scene-classification`

Either clone the GitHub repo after it is created or upload this project zip and unzip it.

```bash
%cd /kaggle/working/spark-ai-scene-classification
!python -m pip install -q onnx onnxruntime
```

## 2. Dataset audit

```bash
!python -m src.audit --data-root /kaggle/input/intel-image-classification
```

## 3. Baseline

```bash
!python -m src.train --data-root /kaggle/input/intel-image-classification --model simple_cnn --epochs 5 --learning-rate 0.001 --batch-size 64 --num-workers 2
```

## 4. Basic tuning

```bash
!python -m src.tune --data-root /kaggle/input/intel-image-classification --learning-rates 0.001 0.0003 0.0001 --epochs-per-trial 2 --batch-size 64 --num-workers 2
```

Read:

```python
import pandas as pd
pd.read_csv('outputs/tuning/tuning_results.csv')
```

## 5. Final MobileNet

Replace `<LR>` with the best tuning learning rate.

```bash
!python -m src.train --data-root /kaggle/input/intel-image-classification --model mobilenet_v3_small --epochs 4 --learning-rate <LR> --batch-size 64 --num-workers 2 --checkpoint models/mobilenet_v3_small_best.pth
```

If four epochs are clearly still improving and time remains, allow one additional epoch. Do not start architecture experiments.

## 6. Evaluate BOTH models

```bash
!python -m src.evaluate --data-root /kaggle/input/intel-image-classification --checkpoint models/simple_cnn_best.pth
!cp outputs/metrics/simple_cnn_test_metrics.json outputs/metrics/baseline_test_metrics.json

!python -m src.evaluate --data-root /kaggle/input/intel-image-classification --checkpoint models/mobilenet_v3_small_best.pth
```

## 7. Error analysis on final model

```bash
!python -m src.error_analysis --predictions outputs/metrics/mobilenet_v3_small_test_predictions.csv --top-n 24
```

## 8. Export final model to ONNX

```bash
!python -m src.export_onnx --checkpoint models/mobilenet_v3_small_best.pth --output models/best_model.onnx
```

## 9. Download artifacts

At minimum preserve:

- `models/best_model.onnx`
- `models/class_names.json`
- `models/mobilenet_v3_small_best.pth` (release artifact, not needed by Render)
- `outputs/metrics/`
- `outputs/plots/`
- `outputs/error_analysis/`
- `outputs/tuning/`

Commit the ONNX model, class names, plots, metrics, and tuning CSV to GitHub. Do not commit the full raw dataset.
