# Neural Network from Scratch: Regression

A single-hidden-layer feedforward neural network implemented from the ground up
in NumPy, with no machine-learning framework. Every core component, such as weight
initialization, the forward pass, backpropagation, and the gradient-descent
update, is written manually.

## What it does
Fits a 1-dimensional regression dataset and uses the network to illustrate
how network capacity and training length must compromise between underfitting and overfitting.

## Implementation details
- **Architecture:** configurable single hidden layer (1 → N → 1)
- **Backpropagation:** gradients derived and computed by hand via the chain
  rule, from the MSE derivative back through the output and hidden layers
- **Activations:** tanh, sigmoid, and ReLU, each with its derivative
- **Initialization:** Xavier (tanh/sigmoid) and He (ReLU)
- **Training modes:** full-batch gradient descent and online SGD
- **Tracking:** train/test MSE recorded per epoch, with learning-curve plots

## Experiments
| Experiment      | Setup                       | Purpose                     |
|-----------------|-----------------------------|-----------------------------|
| Baseline        | 10 neurons, tanh, batch     | Fit the data                |
| Underfitting    | 2 neurons                   | Too little capacity         |
| Optimal         | 10 neurons                  | Balanced fit                |
| Overfitting     | 50 neurons, 20k epochs      | Memorization                |
| Batch vs online | 10 neurons                  | Compare training modes      |
| Activation      | ReLU, 15 neurons            | Compare activation choices  |

Each run plots the fitted curve against the train/test points and the learning
curve (MSE over epochs, log scale).

## Running
```bash
python regression.py
```
The script expects a data file at `dane/dane2.txt` containing whitespace-separated
`x y` pairs. This file is included in the repository.
