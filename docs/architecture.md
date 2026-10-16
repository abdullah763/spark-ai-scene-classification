# Architecture

## Training path

Public Intel scene dataset -> deterministic train/validation split + untouched provided test split -> preprocessing/augmentation -> SimpleCNN baseline and MobileNetV3-Small -> validation selection -> final test evaluation -> error analysis -> ONNX export.

## Serving path

Uploaded JPEG/PNG/WEBP -> validation -> PIL RGB decode -> resize + ImageNet normalization -> ONNX Runtime MobileNetV3-Small -> softmax -> predicted class, confidence, top-3 predictions.

## Engineering choices

- SimpleCNN provides a genuine from-scratch baseline.
- MobileNetV3-Small is the stronger transfer-learning model and is lightweight enough for CPU inference.
- Hyperparameter tuning is intentionally small and controlled because the assessment values engineering judgement under compute/time constraints.
- PyTorch is used for training; ONNX Runtime is used in the public API to reduce deployment memory and dependency size.
- The API starts in a degraded state if model artifacts are absent rather than failing process startup.
