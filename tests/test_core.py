"""
Unit tests for the core functionality of the doodle recognizer application.
These tests cover model loading, prediction probabilities, preprocessing, and error handling.
"""

import unittest
import numpy as np

from PIL import Image, ImageDraw
from predict import load_model, predict_doodle
from preprocessing import preprocess_image


class TestDoodleRecognizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.params, cls.categories, cls.input_size, cls.divisor = load_model()

    def setUp(self):
        self.canvas = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(self.canvas)
        draw.line((20, 20, 80, 80), fill="black", width=5)

    def test_model_loading(self):
        self.assertEqual(len(self.params), 6)
        self.assertEqual(len(self.categories), 10)
        self.assertEqual(self.input_size, 784)
        self.assertEqual(self.divisor, 255.0)

        self.assertEqual(self.params[0].shape[0], 784)
        self.assertEqual(self.params[-1].shape[1], 10)

    def test_prediction_probabilities(self):
        image = preprocess_image(self.canvas)

        category, confidence, probabilities = predict_doodle(
            image, self.params, self.categories
        )

        self.assertEqual(probabilities.shape, (10,))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=5)

        self.assertEqual(
            category,
            self.categories[int(np.argmax(probabilities))]
        )
        self.assertAlmostEqual(
            confidence, float(np.max(probabilities)), places=6
        )

    def test_invalid_prediction_inputs(self):
        invalid_images = [
            np.zeros(783, dtype=np.float32),
            np.full(784, -0.1, dtype=np.float32),
            np.full(784, 1.1, dtype=np.float32),
            np.full(784, np.nan, dtype=np.float32),
        ]

        for image in invalid_images:
            with self.subTest(image_size=image.size, first_pixel=image[0]):
                with self.assertRaises(ValueError):
                    predict_doodle(image, self.params, self.categories)

    def test_preprocessing_output(self):
        result = preprocess_image(self.canvas)

        self.assertEqual(result.shape, (28, 28))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreater(float(result.max()), 0)
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)

    def test_default_preprocessing_unchanged(self):
        default = preprocess_image(self.canvas)

        explicit = preprocess_image(
            self.canvas,
            crop_to_drawing=False,
            enhance_strokes=False
        )

        np.testing.assert_array_equal(default, explicit)

    def test_auto_crop(self):
        result = preprocess_image(
            self.canvas,
            crop_to_drawing=True
        )

        self.assertEqual(result.shape, (28, 28))
        self.assertGreater(float(result.sum()), 0)
        self.assertLessEqual(float(result.max()), 1)

    def test_stroke_thickening(self):
        original = preprocess_image(self.canvas)

        thickened = preprocess_image(
            self.canvas,
            enhance_strokes=True
        )

        self.assertEqual(thickened.shape, original.shape)
        self.assertTrue(np.all(thickened >= original))
        self.assertGreater(float(thickened.sum()), float(original.sum()))

    def test_transparent_background(self):
        transparent = Image.new(
            "RGBA", (100, 100), (0, 0, 0, 0)
        )

        result = preprocess_image(transparent)

        np.testing.assert_array_equal(
            result,
            np.zeros((28, 28), dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
