# Python Machine Learning Projects

A small collection of machine learning projects in Python, spanning deep
learning with Keras/TensorFlow and a neural network implemented from scratch in
pure NumPy.

## Projects

### 1. Neural Network from Scratch — Regression
A single-hidden-layer feedforward network built entirely with NumPy: manual
forward propagation, backpropagation, and gradient descent, with no ML
framework. Used to fit a 1D regression problem and to demonstrate underfitting,
optimal capacity, and overfitting by varying network size and training length.
Supports tanh / sigmoid / ReLU activations, He/Xavier initialization, and both
batch and online training.

→ [`neural-network-from-scratch/`](neural-network-regression/)

### 2. Image Classification with CNNs — CIFAR-10
Three convolutional neural networks of increasing depth, built with
Keras/TensorFlow and compared on the CIFAR-10 dataset. Includes data
augmentation, batch normalization, dropout, and training callbacks (early
stopping, learning-rate reduction), followed by a full evaluation: confusion
matrices, per-class accuracy, and misclassification analysis.

→ [`cnn-image-classification/`](cnn-image-classification/)

## Tech stack
Python · NumPy · TensorFlow/Keras · scikit-learn · Matplotlib · Seaborn

## Setup
```bash
pip install -r requirements.txt
```
Each project folder has its own README with details on how to run it.