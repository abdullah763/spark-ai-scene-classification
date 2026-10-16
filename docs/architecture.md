# Architecture

## Overview

This project separates the ML training workflow from the production inference workflow.

Training is performed in PyTorch on Kaggle GPU resources. The selected MobileNetV3-Small model is exported to ONNX, and the production FastAPI service uses ONNX Runtime for lightweight CPU inference.

## Training Pipeline

```text
Intel Image Classification dataset
        |
        v
Dataset audit
        |
        v
Deterministic train/validation split
+ untouched provided test split
        |
        v
Preprocessing and augmentation
        |
        +----------------------+
        |                      |
        v                      v
SimpleCNN                MobileNetV3-Small
baseline                 transfer learning
        |                      |
        +----------+-----------+
                   |
                   v
        Validation-based selection
                   |
                   v
             Test evaluation
                   |
                   v
       Quantitative error analysis
                   |
                   v
        Qualitative error analysis
                   |
                   v
              ONNX export
```

## Inference Pipeline

```text
JPEG / PNG / WEBP image
        |
        v
FastAPI upload validation
        |
        v
PIL RGB decoding
        |
        v
Resize + ImageNet normalization
        |
        v
ONNX Runtime
MobileNetV3-Small
        |
        v
Softmax probabilities
        |
        v
Predicted class
+ confidence
+ top-3 predictions
```

## Deployment Pipeline

```text
GitHub repository
        |
        v
Docker build
        |
        v
Railway
        |
        v
Public FastAPI endpoint
```

Public deployment:

```text
https://spark-ai-scene-classification-production.up.railway.app
```

## Main Engineering Decisions

### SimpleCNN baseline

A small CNN trained from scratch provides a genuine baseline and verifies that the data pipeline works independently of transfer learning.

### MobileNetV3-Small

MobileNetV3-Small was selected as the stronger model because it provides a good balance of:

- transfer-learning performance
- fast training
- low parameter count
- compact deployment size
- efficient CPU inference

### Validation-based checkpoint selection

The best checkpoint is selected using validation macro-F1 rather than simply taking the final epoch.

### Controlled hyperparameter tuning

A small learning-rate search was used instead of expensive brute-force tuning because the assessment emphasizes practical ML engineering judgment under compute and time constraints.

### Untouched final test split

The provided labeled test set is used only for final evaluation after model selection.

### ONNX Runtime deployment

PyTorch is used for training, while ONNX Runtime is used for production serving.

Benefits:

- reduced production dependency size
- lower memory footprint
- simpler CPU deployment
- approximately 5.82 MB final ONNX model

### Graceful degraded mode

If the ONNX model is missing, the API starts successfully but reports:

```text
model_loaded = false
```

This prevents deployment crashes and makes operational state visible through the health endpoint.

## Project Layers

### Data layer

- `src/audit.py`
- `src/data.py`
- `src/transforms.py`

Responsibilities:

- dataset discovery
- dataset quality audit
- class distribution analysis
- deterministic splitting
- preprocessing
- augmentation

### Model and training layer

- `src/models.py`
- `src/train.py`
- `src/tune.py`

Responsibilities:

- model definitions
- baseline training
- transfer learning
- checkpoint selection
- controlled hyperparameter tuning

### Evaluation layer

- `src/evaluate.py`
- `src/error_analysis.py`

Responsibilities:

- final test metrics
- classification reports
- prediction-level outputs
- confusion matrices
- per-class error analysis
- common confusion pairs
- high-confidence failure inspection
- qualitative failure visualization

### Deployment layer

- `src/export_onnx.py`
- `src/inference.py`
- `api/main.py`
- `Dockerfile`

Responsibilities:

- PyTorch-to-ONNX export
- ONNX numerical validation
- production preprocessing
- ONNX Runtime inference
- request validation
- API health reporting
- public serving

## Model Flow

```text
PyTorch checkpoint
        |
        v
ONNX export
        |
        v
Numerical validation
max_abs_diff = 2.14577e-06
        |
        v
models/best_model.onnx
        |
        v
ONNXSceneClassifier
        |
        v
FastAPI /predict
```

## Final Model Performance

The selected MobileNetV3-Small model achieved:

- Test accuracy: 0.9200
- Macro precision: 0.9219
- Macro recall: 0.9221
- Macro F1: 0.9217

The SimpleCNN baseline achieved:

- Test accuracy: 0.8127
- Macro F1: 0.8147

The largest remaining failure mode is confusion between mountain and glacier scenes.
