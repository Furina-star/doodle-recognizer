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

# Print the split dataset shapes
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Print the class distribution in each split
print("Training classes:", np.bincount(y_train))
print("Validation classes:", np.bincount(y_val))
print("Test classes:", np.bincount(y_test))