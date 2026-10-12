"""
This module provides functions to load a trained neural network model and predict the category of a doodle image.
Functions:
- load_model(model_path): Loads the model parameters, categories, input size, and normalization divisor from a specified .npz file.
- predict_doodle(image, params, categories): Predicts the category of a doodle image using the provided model parameters and categories.
"""

import numpy as np
from pathlib import Path
from ann import forward

# Local path to the project root directory, used for loading model files
PROJECT_ROOT = Path(__file__).resolve().parent

# Load the trained model parameters, categories, input size, and normalization divisor from a .npz file
def load_model(model_path="model/doodle_ann_v1.npz"):
    model_path = Path(model_path)

    # If the provided model path is not absolute, resolve it relative to the project root
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    with np.load(model_path, allow_pickle=False) as saved:
        params = [
            saved["W1"],
            saved["b1"],
            saved["W2"],
            saved["b2"],
            saved["W3"],
            saved["b3"]
        ]

        categories = saved["categories"].tolist()
        input_size = int(saved["input_size"])
        divisor = float(saved["normalization_divisor"])

    return params, categories, input_size, divisor

# Predict the category of a doodle image
def predict_doodle(image, params, categories):
    image = np.asarray(image, dtype=np.float32)

    expected_size = params[0].shape[0]

    if image.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} pixels, got {image.size}"
        )

    if not np.all(np.isfinite(image)) or np.any(image < 0) or np.any(image > 1):
        raise ValueError("Image pixels must be finite values between 0 and 1")

    image_batch = image.reshape(1, expected_size)

    probabilities, _ = forward(image_batch, *params)

    probabilities = probabilities[0]

    predicted_id = int(np.argmax(probabilities))
    predicted_category = categories[predicted_id]
    confidence = float(probabilities[predicted_id])

    return predicted_category, confidence, probabilities

