# Neural Network Architecture

## Overview

Doodle Recognizer uses a fully connected feedforward artificial neural network (ANN) implemented with NumPy. It classifies a normalized 28 × 28 grayscale doodle into one of 10 categories. TensorFlow and PyTorch are not used.

## Layer configuration

| Layer | Input shape | Output shape | Activation |
|---|---|---|---|
| Input | `(N, 784)` | `(N, 784)` | None |
| Hidden 1 | `(N, 784)` | `(N, 64)` | ReLU |
| Hidden 2 | `(N, 64)` | `(N, 32)` | ReLU |
| Output | `(N, 32)` | `(N, 10)` | Softmax |

`N` is the batch size. Each 28 × 28 image is flattened into 784 values in the range `[0, 1]`.

| Parameter | Shape |
|---|---|
| `W1`, `b1` | `(784, 64)`, `(1, 64)` |
| `W2`, `b2` | `(64, 32)`, `(1, 32)` |
| `W3`, `b3` | `(32, 10)`, `(1, 10)` |

The network has **52,650 trainable parameters** (weights and biases).

## Forward propagation

Implemented in `ann.py` as `forward(X, W1, b1, W2, b2, W3, b3)`:

```text
Z1 = X  @ W1 + b1
A1 = ReLU(Z1)
Z2 = A1 @ W2 + b2
A2 = ReLU(Z2)
Z3 = A2 @ W3 + b3
P  = Softmax(Z3)
```

- **ReLU:** `max(0, z)` introduces nonlinearity.
- **Softmax:** `exp(z_i - max(z)) / sum_j exp(z_j - max(z))` returns a probability distribution over the 10 categories. Subtracting the row maximum improves numerical stability.
- **Prediction:** `argmax(P, axis=1)` selects the category with the highest output probability.

`forward()` returns the probability matrix and a cache `(Z1, A1, Z2, A2)` for backpropagation.

## Loss function

`cross_entropy(P, y)` computes the mean categorical cross-entropy:

```text
L = -(1/N) * sum_i log(P[i, y[i]])
```

Probabilities are clipped to `[1e-12, 1]` before taking the logarithm to avoid `log(0)`.

## Backpropagation

`backward()` in `ann.py` uses the chain rule to calculate gradients. With `Y` as one-hot labels:

```text
dZ3 = (P - Y) / N
dW3 = A2.T @ dZ3
db3 = sum(dZ3, axis=0)

dZ2 = (dZ3 @ W3.T) * (Z2 > 0)
dW2 = A1.T @ dZ2
db2 = sum(dZ2, axis=0)

dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
dW1 = X.T @ dZ1
db1 = sum(dZ1, axis=0)
```

The derivative of ReLU is 1 where the pre-activation value is positive and 0 otherwise. `backward()` returns gradients for all six trainable parameter arrays.

## Parameter updates

`train.py` performs mini-batch gradient descent with L2 regularization on weights only:

```text
W = W - learning_rate * (dW + l2_lambda * W)
b = b - learning_rate * db
```

Weight matrices use He initialization; biases start at zero. Model selection, hyperparameters, and dataset splitting are described in [TRAINING.md](TRAINING.md).

## Source files

| File | Responsibility |
|---|---|
| [`ann.py`](../ann.py) | Forward pass, Softmax, cross-entropy, backward pass |
| [`train.py`](../train.py) | Initialization, gradient checking, parameter updates |
| [`predict.py`](../predict.py) | Load saved weights and perform inference |
| [`main.py`](../main.py) | Coordinate training and save the selected checkpoint |

The trained weights are stored in [`model/doodle_ann_v1.npz`](../model/doodle_ann_v1.npz).
