"""
This module implements a simple feedforward neural network with two hidden layers and ReLU activation functions.
The network is designed for multi-class classification tasks and uses the softmax function for the output layer
and categorical cross-entropy loss for training. The network is trained using mini-batch gradient descent
with backpropagation to update the weights and biases.
The forward function computes the output probabilities of the network given an input batch, while the backward function computes the gradients of the loss with respect to the weights and biases for backpropagation.
The cross_entropy function calculates the loss for a batch of predictions and true labels.
"""

import numpy as np

# Forward Propagation
def forward (X, W1, b1, W2, b2, W3, b3):
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)  # ReLU activation

    Z2 = A1 @ W2 + b2
    A2 = np.maximum(0, Z2)  # ReLU activation

    Z3 = A2 @ W3 + b3
    P = softmax(Z3)  # Softmax activation for output layer

    cache = (Z1, A1, Z2, A2)
    return P, cache

# Softmax activation function for the output layer
def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_values = np.exp(shifted)

    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Calculate the categorical cross-entropy loss for the batch
def cross_entropy(P, y):
    correct_probs = P[np.arange(len(y)), y]
    safe_probs = np.clip(correct_probs, 1e-12, 1.0)

    return -np.mean(np.log(safe_probs))

# Backpropagation for the output layer
def backward(X_batch, y_batch, P, cache, W2, W3):
    Z1, A1, Z2, A2 = cache

    # Backpropagation for the output layer
    N = len(y_batch)
    Y = np.eye(P.shape[1], dtype=P.dtype)[y_batch]
    dZ3 = (P - Y) / N

    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    # Backpropagation for the second hidden layer
    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (Z2 > 0)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Backpropagation for the first hidden layer
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X_batch.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3