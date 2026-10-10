import numpy as np
from sklearn.model_selection import train_test_split

apple_data = np.load(
    "data/raw/full_numpy_bitmap_apple.npy",
    mmap_mode="r"
)

cat_data = np.load(
    "data/raw/full_numpy_bitmap_cat.npy",
    mmap_mode="r"
)

car_data = np.load(
    "data/raw/full_numpy_bitmap_car.npy",
    mmap_mode="r"
)

cat_images = cat_data[:5000].astype(np.float32) / 255.0 # Normalize the pixel values to be between 0 and 1
cat_labels = np.full(len(cat_images), 0, dtype=np.int64)

apple_images = apple_data[:5000].astype(np.float32) / 255.0 # Normalize the pixel values to be between 0 and 1
apple_labels = np.full(len(apple_images), 1, dtype=np.int64)

car_images = car_data[:5000].astype(np.float32) / 255.0 # Normalize the pixel values to be between 0 and 1
car_labels = np.full(len(car_images), 2, dtype=np.int64)

# Concatenate the images of cats, apples, and cars into a single array
X = np.concatenate(
    (cat_images, apple_images, car_images),
    axis=0
)

# Concatenate the labels of cats, apples, and cars into a single array
y = np.concatenate(
    (cat_labels, apple_labels, car_labels),
    axis=0
)

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
    Y = np.eye(3, dtype=np.float32)[y_batch]
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

# Initialize trainable parameters
rng = np.random.default_rng(42)

W1 = rng.standard_normal((784, 16)).astype(np.float32)
W1 *= np.sqrt(2 / 784)
b1 = np.zeros((1, 16), dtype=np.float32)

W2 = rng.standard_normal((16, 8)).astype(np.float32)
W2 *= np.sqrt(2 / 16)
b2 = np.zeros((1, 8), dtype=np.float32)

W3 = rng.standard_normal((8, 3)).astype(np.float32)
W3 *= np.sqrt(2 / 8)
b3 = np.zeros((1, 3), dtype=np.float32)

batch_size = 32
epochs = 5
learning_rate = 0.01
for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_samples = 0

    indices = rng.permutation(len(X_train))

    for start in  range(0,len(X_train), batch_size):
        batch_indices = indices[start:start + batch_size]

        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        P, cache = forward(X_batch, W1, b1, W2, b2, W3, b3)
        batch_loss = cross_entropy(P, y_batch)
        predictions = np.argmax(P, axis=1)
        epoch_loss += batch_loss * len(y_batch)
        epoch_correct += np.sum(predictions == y_batch)
        epoch_samples += len(y_batch)
        gradient = backward(X_batch, y_batch, P, cache, W2, W3)

        # Unpack the gradients
        dW1, db1, dW2, db2, dW3, db3 = gradient

        # Update the weights and biases using gradient descent
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    avg_loss = epoch_loss / epoch_samples
    accuracy = epoch_correct / epoch_samples

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {avg_loss:.4f} | "
        f"Accuracy: {accuracy:.2%}"
    )




