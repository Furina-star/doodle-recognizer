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

# First hidden layer
X_batch = X_train[:32] # Select the first 32 samples from the training set as a batch
rng = np.random.default_rng(seed=42)
W1 = rng.standard_normal((784, 16)).astype(np.float32)
W1 *= np.sqrt(2/784)
b1 = np.zeros((1, 16), dtype=np.float32)
Z1 = X_batch @ W1 + b1
A1 = np.maximum(0, Z1)

# Second hidden layer
W2 = rng.standard_normal((16, 8)).astype(np.float32)
W2 *= np.sqrt(2/16)
b2 = np.zeros((1, 8), dtype=np.float32)
Z2 = A1 @ W2 + b2
A2 = np.maximum(0, Z2)

# Output layer
W3 = rng.standard_normal((8, 3)).astype(np.float32)
W3 *= np.sqrt(2/8)
b3 = np.zeros((1, 3), dtype=np.float32)
Z3 = A2 @ W3 + b3

# Softmax activation function for the output layer
def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_values = np.exp(shifted)

    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

probabilities = softmax(Z3)
predictions = np.argmax(probabilities, axis=1)

print("Predicted labels:", predictions[:5])
print("Actual labels:", y_train[:5])

print("Probabilities shape:", probabilities.shape)
print("First drawing: ", probabilities[0])
print("Probabilities sum: \n", probabilities.sum(axis=1))
print("Valid probabilities: ", np.allclose(probabilities.sum(axis=1), 1.0))
