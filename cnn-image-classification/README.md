# Image Classification with CNNs — CIFAR-10

Three convolutional neural networks of increasing depth, built with
Keras/TensorFlow and compared on the CIFAR-10 image dataset (10 classes,
32×32 colour images).

## Approach
- **Data:** CIFAR-10 combined and re-split 60/20/20 (train/validation/test),
  normalized to [0, 1]
- **Augmentation:** rotation, width/height shifts, horizontal flip, zoom
- **Models:** three architectures with 1, 2, and 3 convolutional blocks to study
  how depth affects performance
- **Regularization:** batch normalization and progressive dropout
- **Training:** Adam optimizer, early stopping, and learning-rate reduction on
  plateau

## Evaluation
- Test-accuracy comparison across the three models
- Confusion matrices and per-class accuracy
- Overfitting/underfitting diagnosis from the train/validation gap
- Analysis of the most frequently confused class pairs, with sample
  misclassified images

## Results
Best model: **Model C (3 conv blocks)** — **84,58%** on the
held-out test set.

Generated output plots can be seen in the /results directory.

## Running
```bash
python cnn_classifier.py
```
CIFAR-10 downloads automatically via Keras on first run. Plots are saved as PNG
files in the working directory.
