# Spark AI Natural Scene Classification

AI / Machine Learning Engineer technical assessment project: multiclass natural-scene image classification using the public Intel Image Classification dataset.

> **Status:** implementation scaffold ready; metrics and live deployment URL will be filled after training/evaluation.

## Problem

Classify an input image into one of six scene categories: buildings, forest, glacier, mountain, sea, or street.

## Models

1. **SimpleCNN** trained from scratch as the baseline.
2. **MobileNetV3-Small** initialized with ImageNet weights as the stronger, deployment-oriented model.

## Dataset

Kaggle: Intel Image Classification (`puneet6060/intel-image-classification`). The provided `seg_train` labels are split deterministically into train/validation subsets. The provided labeled `seg_test` directory remains untouched for final testing.

## Project structure

```text
api/                  FastAPI service
src/                  data, training, tuning, evaluation, error analysis, ONNX export
models/               final deployment artifacts
outputs/               metrics, plots, error-analysis artifacts
models/               model artifacts
tests/                API tests
docs/                 architecture documentation
```

## Training workflow

```bash
python -m src.audit --data-root /kaggle/input/intel-image-classification

python -m src.train \
  --data-root /kaggle/input/intel-image-classification \
  --model simple_cnn \
  --epochs 5 \
  --learning-rate 0.001 \
  --batch-size 64

python -m src.tune \
  --data-root /kaggle/input/intel-image-classification \
  --learning-rates 0.001 0.0003 0.0001 \
  --epochs-per-trial 2 \
  --batch-size 64
```

After selecting the best tuning learning rate, train the final MobileNetV3-Small with full-network fine-tuning:

```bash
python -m src.train \
  --data-root /kaggle/input/intel-image-classification \
  --model mobilenet_v3_small \
  --epochs 4 \
  --learning-rate <SELECTED_LR> \
  --batch-size 64 \
  --checkpoint models/mobilenet_v3_small_best.pth
```

## Evaluation and error analysis

```bash
python -m src.evaluate \
  --data-root /kaggle/input/intel-image-classification \
  --checkpoint models/mobilenet_v3_small_best.pth

python -m src.error_analysis \
  --predictions outputs/metrics/mobilenet_v3_small_test_predictions.csv
```

## Export for deployment

```bash
python -m src.export_onnx \
  --checkpoint models/mobilenet_v3_small_best.pth \
  --output models/best_model.onnx
```

Commit `models/best_model.onnx` and `models/class_names.json` so the hosting service can load them.

## API

Run locally:

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /`
- `GET /health`
- `POST /predict`
- `GET /docs`

Example:

```bash
curl -X POST -F "file=@sample.jpg" http://127.0.0.1:8000/predict
```

## Live deployment

**URL:** TBD

Render configuration is version-controlled in `render.yaml`. Python is pinned to 3.11.11 to avoid relying on the hosting platform's moving default runtime.

## Results

TBD after the controlled training runs.

## Error analysis

TBD after final test evaluation. The pipeline writes per-class error rates, common confusion pairs, confidence statistics, and a visual grid of the highest-confidence mistakes.

## Limitations and next steps

TBD after observing real model failures. With additional compute, likely directions include longer fine-tuning, augmentation tuning, a larger efficient backbone, higher-resolution inputs, and systematic calibration/robustness analysis.
