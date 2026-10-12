# Image Preprocessing and Gradio Inference

## Input formats

Training and test images are already 28 × 28 Quick Draw bitmaps. `data_loader.py` converts their pixel values to `float32` and divides by 255. It does **not** apply the Gradio image-preprocessing pipeline to these bitmaps.

Browser drawings require conversion before inference. `preprocessing.py` provides `preprocess_image(image, crop_to_drawing=False, enhance_strokes=False)`.

## Browser image pipeline

For each PIL image from Gradio, the function:

1. Converts the image to RGBA and composites transparency over a white background.
2. Converts the result to grayscale.
3. Optionally crops and centers the visible drawing.
4. Resizes the grayscale image to **28 × 28** using Lanczos resampling.
5. Inverts pixel intensity so dark strokes become high values, matching the model's input convention.
6. Optionally thickens strokes with a 3 × 3 maximum filter.
7. Converts pixels to `float32` and divides by 255, returning a `(28, 28)` array in `[0, 1]`.

`predict_doodle()` then flattens the processed image to `(1, 784)` and runs a forward pass through the saved ANN.

## Optional auto-crop and center

When `crop_to_drawing=True`:

- Ink is detected where inverted grayscale intensity is greater than `25` (on a 0–255 scale).
- The smallest bounding box containing detected ink is extracted.
- The content is centered in a white square with padding of `max(2, round(0.12 * side))` pixels on each side, where `side` is the larger bounding-box dimension.
- The square is resized to 28 × 28 during the standard pipeline.

If no ink is detected, the crop operation is skipped. The Gradio checkbox **Auto-crop and center drawing** defaults to ON. Auto-cropping can help off-center drawings, but it is not guaranteed to improve every prediction.

## Experimental stroke thickening

When `enhance_strokes=True`, the processed 28 × 28 image is modified with Pillow `ImageFilter.MaxFilter(3)`. This is a **full 3 × 3 maximum filter**, which expands bright strokes. It is not an adaptive thin-stroke detector or a 25%-strength blend.

The Gradio checkbox **Thicken strokes (experimental)** defaults to OFF. In a four-drawing, paired manual experiment:

| Drawing | OFF | ON |
|---|---|---|
| Cat, thin brush | Sun 99% (wrong) | Sun 98% (wrong) |
| Cat, thick brush | Cat 79% (correct) | Cat 100% (correct) |
| Fish, thin brush | Sun 96% (wrong) | Fish 51% (correct) |
| Fish, thick brush | Fish 27% (correct) | Car 100% (wrong) |

Both configurations classified **2 of 4** examples correctly. The results are illustrative, not a reliable population-level accuracy estimate. Stroke thickening remains optional because it can both help and hurt.

## Gradio inference workflow

[`app.py`](../app.py) loads the saved model once at startup and displays a Sketchpad, predictions, and an enlarged 28 × 28 preview. The Sketchpad uses a black brush on a white canvas; its default brush size is 20.

The recognition button uses `preprocess=False` to receive raw Sketchpad editor data. The application reads the composite PNG rather than relying on Gradio's layer preprocessing, a workaround for a previously observed intermittent `layer_0.png` error. The loader checks that the resolved file path falls within the configured Gradio temporary directory (or its default location), requires a `.png` extension, and opens the image using Pillow. The app rejects blank drawings before inference.

The 28 × 28 processed image is enlarged using nearest-neighbor interpolation for display; enlargement does not affect the ANN input.

## Limitations

- The ANN was trained on Quick Draw bitmaps, while live drawings may differ in style, scale, and line thickness. This distribution difference can reduce accuracy.
- Thickening strokes can merge details and produce confidently incorrect classifications.
- The auto-crop and thickening options were explored through small manual experiments; neither is a proven general improvement.
- The Gradio composite-file workaround was developed for the tested application environment. Test it again before exposing the app publicly.

See [EVALUATION.md](EVALUATION.md) for the final 7,500-image Quick Draw test results.
