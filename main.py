import numpy as np

inputs = np.array([0.6, 0.2])
weight = np.array([0.4, 0.8])
bias = 0.05

z = np.dot(inputs, weight) + bias
print("Weighted sum:", z)

def relu(z):
    return np.maximum(0, z)

activation = relu(z) # Calculate the activation using the ReLU function

target = 1
learning_rate = 0.01 # Fixed learning rate can be modified to a dynamic one if needed

delta = (activation - target) * (z > 0) # Loss gradient with respect to z
weight_gradients = inputs * delta # The Weight gradients are calculated from the inputs(x) and the delta (error/loss)
bias_gradient = delta # The bias gradient is simply the delta (error/loss)

weight -= learning_rate * weight_gradients # Get the weight(w) subtract it to (learning_rate * weight_gradients) to update the weight(w)
bias -= learning_rate * bias_gradient # Get the bias(b) subtract it to (learning_rate * bias_gradient) to update the bias(b)

new_z = np.dot(inputs, weight) + bias # Recalculate the weighted sum after updating weights and bias
new_activation = relu(new_z) # Calculate the new activation using the ReLU function after updating weights and bias

print("Before update:", activation)
print("After update:", new_activation)

old_loss = 0.5 * (activation - target) ** 2 # Calculate the old loss using the mean squared error formula: 0.5 * (activation - target) ** 2
new_loss = 0.5 * (new_activation - target) ** 2 # Calculate the new loss using the mean squared error formula: 0.5 * (new_activation - target) ** 2

print("Old loss:", old_loss)
print("New loss:", new_loss)

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

print("==========================================")
print("First image label:", y[0])
print("Last image label:", y[-1])

print("Image count:", len(X))
print("Label shape:", y.shape)

print("Cat images shape:", cat_images.shape)
print("Cat labels shape:", cat_labels.shape)

print("Apple images shape:", apple_images.shape)
print("Apple labels shape:", apple_labels.shape)

print("Car images shape:", car_images.shape)
print("Car labels shape:", car_labels.shape)

print("First 10 labels:", y[:10])
print("Unique labels:", np.unique(y, return_counts=True))