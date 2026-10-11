"""
This module provides a function to preprocess images for machine learning models.
The preprocessing steps include:
1. Converting the image to RGBA format.
2. Compositing the image onto a white background to handle transparency.
3. Converting the image to grayscale.
4. Resizing the image to 28x28 pixels.
5. Inverting the pixel values.
The final output is a normalized numpy array suitable for model input.
"""

import numpy as np
from PIL import Image, ImageOps

# Preprocess an image for model input
def preprocess_image(image, crop_to_drawing=False):

    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image")

    rgba = image.convert("RGBA")

    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)

    grayscale = background.convert("L")

    if crop_to_drawing:
        ink = np.asarray(ImageOps.invert(grayscale)) > 25

        if np.any(ink):
            ys, xs = np.nonzero(ink)

            content = grayscale.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

            side = max(content.size)
            padding = max(2, int(round(side * 0.12)))

            square_size = side + 2 * padding
            square = Image.new("L", (square_size, square_size), 255)

            position = ((square_size - content.width) // 2, (square_size - content.height) // 2)

            square.paste(content, position)
            grayscale = square


    resized = grayscale.resize((28, 28), Image.Resampling.LANCZOS)

    inverted = ImageOps.invert(resized)

    normalized = np.asarray(inverted, dtype=np.float32) / 255.0

    return   normalized