import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.ticker import PercentFormatter
from pathlib import Path

# Load the dataset from data/raw/
categories = ["cat", "apple", "car"]
sample_per_class = 5000
num_classes = len(categories)
data_directory = Path("data/raw")

# Container store data in a lists
image_parts = []
label_parts = []

# Load images and labels for each category
for label, category in enumerate(categories):
    filename = f"full_numpy_bitmap_{category}.npy"
    file_path = data_directory / filename

    data = np.load(file_path, mmap_mode="r")

    images = data[:sample_per_class].astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    labels = np.full(len(images), label, dtype=np.int64)  # Create labels for the images

    image_parts.append(images)
    label_parts.append(labels)

# Concatenate all images and labels into single arrays
X = np.concatenate(image_parts, axis=0)
y = np.concatenate(label_parts, axis=0)

# Dataset information
print("Categories:", categories)
print("Total images:", X.shape)
print("Labels:", y.shape)
print("Samples per class:", np.bincount(y))

print("Dataset information verified successfully.\n")

assert X.shape[1] == 784
assert len(X) == len(y)
assert len(np.unique(y)) == num_classes


# Randomly shuffle the dataset to ensure that the order of the images and labels is mixed
rng = np.random.default_rng(seed=42) # Create a random number generator with a fixed seed for reproducibility
indices = rng.permutation(len(X)) # Generate a random permutation of indices based on the length of the dataset

X = X[indices] # Shuffle the images using the random indices
y = y[indices] # Shuffle the labels using the same random indices

# Data splitting: Training, Validation, and Testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    stratify= y,
    random_state= 42,
)

# Split the remaining 30% of the data into validation and test sets (15% each)
X_val, X_test, y_val, y_test= train_test_split(
    X_temp , y_temp,
    test_size= 0.5,
    stratify= y_temp,
    random_state= 42,
)

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

# Initialize trainable parameters
rng_weights = np.random.default_rng(42)
rng_batches = np.random.default_rng(123)

# Set hyperparameters for training
batch_size = 32 # Batch size for training
epochs = 30 # Number of epochs to train the model
patience = 5 # Number of epochs to wait for improvement before early stopping
epoch_without_improvement = 0 # Counter for epochs without improvement
learning_rate = 0.1 # Learning rate for gradient descent
hidden1 = 64 # Number of neurons in the first hidden layer
hidden2 = 32 # Number of neurons in the second hidden layer

# Initialize variables to track the best validation loss and corresponding epoch and parameters
best_val_loss = float("inf")
best_epoch = 0
best_params = None

input_size = X_train.shape[1]
W1 = rng_weights.standard_normal((input_size, hidden1)).astype(np.float32)
W1 *= np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden1), dtype=np.float32)

W2 = rng_weights.standard_normal((hidden1, hidden2)).astype(np.float32)
W2 *= np.sqrt(2 / hidden1)
b2 = np.zeros((1, hidden2), dtype=np.float32)

W3 = rng_weights.standard_normal((hidden2, num_classes)).astype(np.float32)
W3 *= np.sqrt(2 / hidden2)
b3 = np.zeros((1, num_classes), dtype=np.float32)

# Gradient Checking (the call is what makes it print)
gradient_check([W1, b1, W2, b2, W3, b3], X_train[:8], y_train[:8])

# Initialize history dictionary to store training and validation loss and accuracy
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}

# Training loop
print ("\nStarting training...")
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

# Verify that the best parameters yield the same validation loss as recorded
P_best, _ = forward(X_val, *best_params)
y_pred = np.argmax(P_best, axis=1)
verified_best_loss = cross_entropy(P_best, y_val)
assert np.isclose(verified_best_loss, best_val_loss)  # confirms best_params matches the saved loss

print(f"\nBest Epoch: {best_epoch}")
print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Acc: {best_val_acc:.2%}\n")

epoch_completed = len(history["train_loss"])
epoch_numbers = range(1, epoch_completed + 1)

# Confusion matrix
cm = np.zeros((num_classes, num_classes), dtype=int)

print("Confusion Matrix:")
for actual, predicted in zip(y_val, y_pred):
    cm[actual, predicted] += 1
print(cm)

correct_per_class = np.diag(cm)
total_per_class = cm.sum(axis=1)
recall = correct_per_class / total_per_class

assert cm.sum() == len(y_val)
cm_accuracy = np.trace(cm) / cm.sum()
assert np.isclose(cm_accuracy, best_val_acc)

class_names = [category.title() for category in categories]

print(f"\nOverall Accuracy: {cm_accuracy:.2%}")
for name, value in zip(class_names, recall):
    print(f"{name} Recall: {value:.2%}")
print("Total predictions:", cm.sum())

# Plot the confusion matrix
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
display.plot(cmap="Blues", values_format="d")
plt.title("Doodle Recognizer - Confusion Matrix")
plt.show()

# Plot the training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(epoch_numbers, history["train_loss"], label="Training Loss")
plt.plot(epoch_numbers, history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Doodle Recognizer - Loss Curves")

plt.legend()
plt.grid(True)
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(epoch_numbers, history["train_acc"], label="Training Accuracy")
plt.plot(epoch_numbers, history["val_acc"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Doodle Recognizer - Accuracy Curves")

plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.xticks(epoch_numbers)

plt.legend()
plt.grid(True)
plt.show()

