# Spark AI Natural Scene Classification

Computer Vision focused **AI / Machine Learning Engineer technical assessment** for Spark AI.

This project implements an end-to-end multiclass image-classification workflow using the public **Intel Image Classification** dataset. It includes dataset auditing, two-model comparison, controlled hyperparameter tuning, test evaluation, quantitative and qualitative error analysis, ONNX export, FastAPI inference, Docker support, automated API tests, and public deployment.

## Live Deployment

- **API:** https://spark-ai-scene-classification-production.up.railway.app
- **Swagger UI:** https://spark-ai-scene-classification-production.up.railway.app/docs
- **Health:** https://spark-ai-scene-classification-production.up.railway.app/health

The production service uses **ONNX Runtime** instead of PyTorch to keep inference lightweight.

---

## Problem

Given a natural-scene image, classify it into one of six categories:

- buildings
- forest
- glacier
- mountain
- sea
- street

The task is non-trivial because some categories have strong visual overlap, especially:

- mountain vs glacier
- buildings vs street
- sea vs glacier

---

## Dataset

**Dataset:** Intel Image Classification  
**Kaggle slug:** `puneet6060/intel-image-classification`

The labeled `seg_train` split was used as the training source and divided deterministically into training and validation subsets. The provided labeled `seg_test` split was kept untouched until final evaluation.

### Dataset Audit

| Split | Images |
|---|---:|
| Training source | 14,034 |
| Final test | 3,000 |
| Total labeled images | 17,034 |

Training-source class distribution:

| Class | Images |
|---|---:|
| buildings | 2,191 |
| forest | 2,271 |
| glacier | 2,404 |
| mountain | 2,512 |
| sea | 2,274 |
| street | 2,382 |

Audit findings:

- **0 corrupt images**
- All audited images were RGB
- Most images were `150 × 150`
- A small number had different aspect ratios
- Class imbalance was mild, so class weighting was not used
- Macro-F1 was used alongside accuracy to keep evaluation class-aware

![Class distribution](outputs/plots/class_distribution.png)

---

## Data Preparation

The pipeline includes:

- RGB conversion
- resizing to model input size
- random augmentation for training
- deterministic preprocessing for validation and test
- ImageNet normalization for MobileNetV3
- deterministic train/validation splitting
- untouched final test evaluation

This keeps model comparison fair and prevents test-set leakage.

---

## Models

### 1. SimpleCNN — Baseline

A lightweight convolutional neural network trained from scratch.

Purpose:

- establish a genuine non-pretrained baseline
- verify the data pipeline independently of transfer learning
- provide a defensible comparison against the stronger model

### 2. MobileNetV3-Small — Final Model

An ImageNet-pretrained `MobileNetV3-Small` model was selected because it provides:

- strong transfer-learning performance
- low parameter count
- fast training
- efficient CPU inference
- a small deployment artifact

---

## Hyperparameter Tuning

A deliberately small controlled search was used because the assessment emphasizes practical engineering decisions rather than brute-force experimentation.

The MobileNetV3 backbone was initially frozen and three learning rates were compared for two epochs:

| Learning Rate | Best Validation Macro-F1 |
|---:|---:|
| **0.001** | **0.9016** |
| 0.0003 | 0.8944 |
| 0.0001 | 0.8828 |

`0.001` was selected for the final training run.

The final MobileNetV3-Small run reached:

**Best validation macro-F1: 0.9217**

![Training curves](outputs/plots/training_curves.png)

---

## Final Test Results

The final evaluation used the untouched 3,000-image test split.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| SimpleCNN | 0.8127 | 0.8234 | 0.8144 | 0.8147 |
| **MobileNetV3-Small** | **0.9200** | **0.9219** | **0.9221** | **0.9217** |

MobileNetV3-Small improved macro-F1 by approximately **10.7 percentage points** over the from-scratch baseline.

### Final Model Error Rate by Class

| Class | Error Rate |
|---|---:|
| forest | 1.48% |
| buildings | 5.49% |
| sea | 5.69% |
| street | 8.78% |
| glacier | 11.39% |
| mountain | 13.90% |

Forest was the strongest class, while mountain and glacier were the most difficult because of their strong visual similarity.

![MobileNetV3 confusion matrix](outputs/plots/mobilenet_v3_small_confusion_matrix.png)

![Per-class error rate](outputs/plots/error_rate_by_class.png)

---

## Error Analysis

The final MobileNetV3-Small model produced:

- **3,000** test predictions
- **2,760** correct predictions
- **240** errors
- **8.0%** overall error rate
- mean confidence on correct predictions: **0.9524**
- mean confidence on incorrect predictions: **0.7465**
- **59** incorrect predictions with confidence >= 0.90

### Most Common Confusion Pairs

| True Class | Predicted Class | Count |
|---|---|---:|
| mountain | glacier | 63 |
| glacier | mountain | 43 |
| street | buildings | 38 |
| buildings | street | 22 |
| sea | glacier | 16 |
| glacier | sea | 13 |

The main failure modes are visually reasonable:

- snowy mountain scenes can resemble glaciers
- glacier scenes containing water can resemble sea scenes
- street scenes often contain buildings as dominant visual features
- building scenes can contain strong street context

![Common failure modes](outputs/error_analysis/common_failure_modes.png)

### High-Confidence Misclassifications

High-confidence mistakes were inspected separately because they are useful for identifying model blind spots and potential dataset issues.

![Highest-confidence misclassifications](outputs/error_analysis/highest_confidence_misclassifications.png)

Manual inspection also revealed at least one potentially inconsistent test label where an urban vehicle scene appears under the `glacier` class. This demonstrates why qualitative inspection is important alongside aggregate metrics.

---

## ONNX Export

The selected MobileNetV3-Small checkpoint was exported from PyTorch to ONNX for deployment.

Deployment artifact:

```text
models/best_model.onnx
```

Approximate size:

```text
5.82 MB
```

PyTorch vs ONNX numerical validation produced:

```text
max_abs_diff = 2.14577e-06
```

The very small difference indicates effectively equivalent outputs between the original PyTorch model and the exported ONNX model.

---

## API

The FastAPI service exposes:

```text
GET  /
GET  /health
GET  /docs
POST /predict
```

Run locally:

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST \
  -F "file=@sample.jpg" \
  http://127.0.0.1:8000/predict
```

Example response structure:

```json
{
  "predicted_class": "forest",
  "confidence": 0.97,
  "top3": [
    {"class": "forest", "confidence": 0.97},
    {"class": "mountain", "confidence": 0.02},
    {"class": "glacier", "confidence": 0.01}
  ]
}
```

The model is loaded once when the application starts.

If the model artifacts are missing, the service starts in a degraded state instead of crashing.

---

## Deployment Architecture

```text
Intel Image Dataset
        |
        v
PyTorch training on Kaggle GPU
        |
        v
MobileNetV3-Small
        |
        v
ONNX export
        |
        v
FastAPI + ONNX Runtime
        |
        v
Docker
        |
        v
Railway public deployment
```

PyTorch is intentionally excluded from the production inference service to reduce deployment size and memory requirements.

---

## Project Structure

```text
api/
    main.py                 FastAPI application

src/
    audit.py                Dataset auditing
    config.py               Shared configuration
    data.py                 Dataset loading and splitting
    transforms.py           Preprocessing and augmentation
    models.py               SimpleCNN and MobileNetV3
    train.py                Training pipeline
    tune.py                 Controlled hyperparameter search
    evaluate.py             Test evaluation
    error_analysis.py       Quantitative/qualitative failure analysis
    inference.py            ONNX Runtime inference
    export_onnx.py          PyTorch -> ONNX export
    utils.py                Utilities

models/
    best_model.onnx
    class_names.json

outputs/
    metrics/
    plots/
    error_analysis/
    tuning/

tests/
    test_api.py

docs/
    architecture.md

Dockerfile
requirements.txt
requirements-api.txt
```

---

## Testing

Run:

```bash
pytest -q
```

Current API test result:

```text
6 passed
```

The tests cover core API health and request-validation behavior.

---

## Engineering Decisions

- Used a real multiclass public dataset rather than a trivial benchmark
- Preserved an untouched final test split
- Compared a from-scratch CNN against transfer learning
- Used a small controlled hyperparameter search instead of expensive brute-force tuning
- Selected checkpoints using validation macro-F1
- Performed both quantitative and qualitative error analysis
- Exported the final model to ONNX for lightweight CPU inference
- Kept training dependencies separate from production-serving dependencies
- Used Docker for reproducible deployment
- Designed the API to degrade gracefully if model artifacts are unavailable

---

## Limitations

- Visually overlapping classes remain difficult, especially mountain vs glacier
- A small number of labels may be noisy or semantically ambiguous
- Hyperparameter tuning was intentionally limited because of assessment time and compute constraints
- Images are relatively low resolution
- Confidence calibration was analyzed but not explicitly optimized

---

## Future Improvements

With additional time and compute, I would explore:

1. longer controlled fine-tuning with early stopping
2. stronger augmentation policies
3. higher-resolution inputs
4. EfficientNet or a larger MobileNet variant
5. probability calibration
6. systematic review of suspected noisy labels
7. robustness testing under blur, lighting, crop, and distribution shifts
8. experiment tracking and CI/CD for automated model validation

---

## Reproducibility

Training was performed on a Kaggle Tesla T4 GPU.

The repository includes:

- deterministic split logic
- configurable parameters
- saved tuning results
- training histories
- model comparison metrics
- per-class classification reports
- prediction-level test outputs
- error-analysis artifacts
- ONNX deployment files

The project intentionally prioritizes **clean ML methodology, explainability of decisions, reproducibility, and deployment readiness** over chasing state-of-the-art benchmark accuracy.
