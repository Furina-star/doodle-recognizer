import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_datasets(categories, sample_per_class=5000, data_directory="data/raw"):
    data_directory = Path(data_directory)
    num_classes = len(categories)

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
    rng = np.random.default_rng(seed=42)  # Create a random number generator with a fixed seed for reproducibility
    indices = rng.permutation(len(X))  # Generate a random permutation of indices based on the length of the dataset

    X = X[indices]  # Shuffle the images using the random indices
    y = y[indices]  # Shuffle the labels using the same random indices

    # Data splitting: Training, Validation, and Testing
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    # Split the remaining 30% of the data into validation and test sets (15% each)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )
