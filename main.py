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

data = np.load(
    "data/raw/full_numpy_bitmap_cat.npy",
    mmap_mode="r"
)

print("Dataset shape:", data.shape)
print("First image shape:", data[0].shape)

image = data[100].reshape(28, 28)  # Reshape the first image to 28x28
normalized = image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

print("Normalized range:", normalized.min(), normalized.max())

import matplotlib.pyplot as plt

plt.imshow(normalized, cmap="gray")
plt.title("Cat Drawing #100")
plt.axis("off")
plt.show()