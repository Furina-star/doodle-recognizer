"""
This module contains functions for training a neural network model.
It includes gradient checking, parameter initialization, and the training loop with early stopping and checkpointing.
"""

import numpy as np
from ann import forward, backward, cross_entropy

# Gradient Checking
PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
def gradient_check(params, X, y, n_samples=5, epsilon=1e-5, tol=1e-4, seed=0):
    # float64 everywhere: float32 is too noisy for epsilon = 1e-5
    params = [p.astype(np.float64).copy() for p in params]
    X = X.astype(np.float64)

    # Analytical gradients: one forward + one backward pass for ALL params
    P, cache = forward(X, *params)
    grads = backward(X, y, P, cache, params[2], params[4])  # needs W2, W3

    def loss():
        P, _ = forward(X, *params)
        return cross_entropy(P, y)

    rng = np.random.default_rng(seed)
    results = {}

    for name, param, grad in zip(PARAM_NAMES, params, grads):
        worst_error = 0.0

        # Check a few random entries per parameter (all of them is too slow)
        flat_indices = rng.choice(param.size, size=min(n_samples, param.size), replace=False)
        for flat_idx in flat_indices:
            idx = np.unravel_index(flat_idx, param.shape)
            original = param[idx]

            param[idx] = original + epsilon
            loss_plus = loss()
            param[idx] = original - epsilon
            loss_minus = loss()
            param[idx] = original  # always restore

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical = grad[idx]
            error = abs(analytical - numerical) / max(1e-8, abs(analytical) + abs(numerical))
            worst_error = max(worst_error, error)

        results[name] = worst_error
        status = "PASSED" if worst_error < tol else "FAILED"
        print(f"{name}: worst relative error = {worst_error:.2e}: {status}")

    return results

# Initialize parameters with He initialization
def initialize_parameters(input_size, hidden1, hidden2, num_classes, rng_weights):
    W1 = rng_weights.standard_normal((input_size, hidden1)).astype(np.float32)
    W1 *= np.sqrt(2 / input_size)
    b1 = np.zeros((1, hidden1), dtype=np.float32)

    W2 = rng_weights.standard_normal((hidden1, hidden2)).astype(np.float32)
    W2 *= np.sqrt(2 / hidden1)
    b2 = np.zeros((1, hidden2), dtype=np.float32)

    W3 = rng_weights.standard_normal((hidden2, num_classes)).astype(np.float32)
    W3 *= np.sqrt(2 / hidden2)
    b3 = np.zeros((1, num_classes), dtype=np.float32)

    return W1, b1, W2, b2, W3, b3

# Train the model with early stopping and checkpointing
def train_model(X_train, y_train,X_val, y_val,params,rng_batches,batch_size,epochs,patience,learning_rate,l2_lambda):
    W1, b1, W2, b2, W3, b3 = params

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_params = None
    epoch_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        indices = rng_batches.permutation(len(X_train))

        for start in range(0, len(X_train), batch_size):
            batch_indices = indices[start:start + batch_size]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            P, cache = forward(X_batch, W1, b1, W2, b2, W3, b3)
            batch_loss = cross_entropy(P, y_batch)
            predictions = np.argmax(P, axis=1)
            epoch_loss += batch_loss * len(y_batch)
            epoch_correct += np.sum(predictions == y_batch)
            epoch_samples += len(y_batch)

            # Unpack the gradients
            dW1, db1, dW2, db2, dW3, db3 = backward(X_batch, y_batch, P, cache, W2, W3)

            # l2 regularization
            dW1 += l2_lambda * W1
            dW2 += l2_lambda * W2
            dW3 += l2_lambda * W3

            # Update the weights and biases using gradient descent
            W3 -= learning_rate * dW3
            b3 -= learning_rate * db3
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1

        avg_loss = epoch_loss / epoch_samples
        accuracy = epoch_correct / epoch_samples

        # Validation pass
        P_val, _ = forward(X_val, W1, b1, W2, b2, W3, b3)
        val_loss = cross_entropy(P_val, y_val)
        val_predictions = np.argmax(P_val, axis=1)
        val_accuracy = np.mean(val_predictions == y_val)

        # Checkpointing: save a copy of the parameters if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            best_params = [
                W1.copy(), b1.copy(),
                W2.copy(), b2.copy(),
                W3.copy(), b3.copy()
            ]
            epoch_without_improvement = 0
        else:
            epoch_without_improvement += 1

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(accuracy)
        history["val_acc"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Train Acc: {accuracy:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2%}"
        )

        # Early stopping check
        if epoch_without_improvement >= patience:
            print(f"Early stopping at Epoch {epoch + 1}. ", f"Best epoch was {best_epoch}.")
            break

    return best_params, history, best_epoch, best_val_loss, best_val_acc

