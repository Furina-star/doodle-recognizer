import numpy as np
from PIL import Image, ImageOps

def preprocess_image(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image")

    rgba = image.convert("RGBA")

    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)

    grayscale = background.convert("L")

    resized = grayscale.resize((28, 28), Image.Resampling.LANCZOS)

    inverted = ImageOps.invert(resized)

    normalized = np.asarray(inverted, dtype=np.float32) / 255.0

    return   normalized