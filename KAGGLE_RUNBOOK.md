# Kaggle Runbook

This runbook documents the compact sequence used to reproduce the training and evaluation workflow for the Spark AI scene-classification assessment.

## Dataset

Use the Kaggle dataset:

```text
puneet6060/intel-image-classification
```

In the current Kaggle mount layout used for this assessment, the dataset root was:

```text
/kaggle/input/datasets/puneet6060/intel-image-classification
```

If Kaggle mounts the dataset differently, locate `seg_train` and `seg_test` first and update `DATA_ROOT`.

---

## 1. Environment and Dataset Check

```python
from pathlib import Path
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

INPUT_ROOT = Path("/kaggle/input")

for p in INPUT_ROOT.rglob("seg_train"):
    print("Found seg_train at:", p)

for p in INPUT_ROOT.rglob("seg_test"):
    print("Found seg_test at:", p)
```

Expected accelerator for this run:

```text
Tesla T4
```

---

## 2. Clone Repository

```bash
git clone https://github.com/abdullah763/spark-ai-scene-classification.git
cd /kaggle/working/spark-ai-scene-classification
```

---

## 3. Dataset Audit

```bash
python -m src.audit \
  --data-root /kaggle/input/datasets/puneet6060/intel-image-classification
```

Audit output is saved under:

```text
outputs/metrics/
outputs/plots/
```

Observed dataset statistics:

```text
Training source: 14,034 images
Final test:      3,000 images
Corrupt images:  0
```

The deterministic split used during training produced:

```text
Training:   11,928
Validation:  2,106
Test:        3,000
```

---

## 4. Train Baseline SimpleCNN

```bash
python -m src.train \
  --data-root /kaggle/input/datasets/puneet6060/intel-image-classification \
  --model simple_cnn \
  --epochs 5 \
  --learning-rate 0.001 \
  --batch-size 64 \
  --num-workers 2
```

Best observed validation macro-F1:

```text
0.8147
```

---

## 5. Tune MobileNetV3-Small

```bash
python -m src.tune \
  --data-root /kaggle/input/datasets/puneet6060/intel-image-classification \
  --learning-rates 0.001 0.0003 0.0001 \
  --epochs-per-trial 2 \
  --batch-size 64 \
  --num-workers 2
```

Observed tuning results:

```text
0.001  -> 0.9016 validation macro-F1
0.0003 -> 0.8944
0.0001 -> 0.8828
```

Selected learning rate:

```text
0.001
```

---

## 6. Train Final MobileNetV3-Small

```bash
python -m src.train \
  --data-root /kaggle/input/datasets/puneet6060/intel-image-classification \
  --model mobilenet_v3_small \
  --epochs 4 \
  --learning-rate 0.001 \
  --batch-size 64 \
  --num-workers 2 \
  --checkpoint models/mobilenet_v3_small_best.pth
```

Best observed validation macro-F1:

```text
0.9217
```

---

## 7. Evaluate Both Models

```bash
python -m src.evaluate \
  --data-root /kaggle/input/datasets/puneet6060/intel-image-classification \
  --checkpoint models/simple_cnn_best.pth \
  --batch-size 64 \
  --num-workers 2

python -m src.evaluate \
  --data-root /kaggle/input/datasets/puneet6060/intel-image-classification \
  --checkpoint models/mobilenet_v3_small_best.pth \
  --batch-size 64 \
  --num-workers 2
```

Final test results:

```text
SimpleCNN:
accuracy  = 0.8127
macro-F1  = 0.8147

MobileNetV3-Small:
accuracy  = 0.9200
macro-F1  = 0.9217
```

---

## 8. Error Analysis

```bash
python -m src.error_analysis \
  --predictions outputs/metrics/mobilenet_v3_small_test_predictions.csv \
  --top-n 20
```

Key observed failure modes:

```text
mountain -> glacier
glacier -> mountain
street -> buildings
buildings -> street
```

The repository includes both quantitative CSV outputs and qualitative error-analysis figures.

---

## 9. Export Final Model to ONNX

Install export/runtime dependencies if needed:

```bash
pip install onnx onnxruntime
```

Export:

```bash
python -m src.export_onnx \
  --checkpoint models/mobilenet_v3_small_best.pth \
  --output models/best_model.onnx
```

Observed export validation:

```text
max_abs_diff = 2.14577e-06
```

Deployment artifacts:

```text
models/best_model.onnx
models/class_names.json
```

---

## Deployment

The final public service is deployed on **Railway** using Docker.

Public API:

```text
https://spark-ai-scene-classification-production.up.railway.app
```

Health endpoint:

```text
https://spark-ai-scene-classification-production.up.railway.app/health
```

Swagger UI:

```text
https://spark-ai-scene-classification-production.up.railway.app/docs
```

The production service uses ONNX Runtime and does not require PyTorch.

---

## Final Notes

The assessment workflow intentionally favors:

- reproducible data splits
- clean model comparison
- controlled tuning
- untouched final test evaluation
- qualitative and quantitative error analysis
- lightweight production inference
- deployment readiness

No additional retraining is required for the submitted version.
