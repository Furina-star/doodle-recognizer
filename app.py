"""
A simple Gradio app for recognizing doodles.
"""

import gradio as gr
import numpy as np
import tempfile

from pathlib import Path
from PIL import Image
from predict import load_model, predict_doodle
from preprocessing import preprocess_image

# Load the model parameters and categories
params, categories, input_size, divisor = load_model()

# Gradio interface function to load the composite image from the editor data
def load_composite_image(editor_data):
    if not isinstance(editor_data, dict):
        raise gr.Error("Please draw something first.")

    composite = editor_data.get("composite")

    if not isinstance(composite, dict):
        raise gr.Error("The drawing image is unavailable.")

    file_path = composite.get("path")

    if not isinstance(file_path, str):
        raise gr.Error("Invalid drawing data.")

    path = Path(file_path).resolve()
    gradio_temp = (Path(tempfile.gettempdir()) / "gradio").resolve()

    if not path.is_relative_to(gradio_temp):
        raise gr.Error("Unexpected drawing file location.")

    try:
        with Image.open(path) as image:
            image.load()
            return image.copy()

    except OSError:
        raise gr.Error(
            "The drawing could not be read. "
            "Please clear the canvas and draw again."
        ) from None


# Gradio interface function to recognize the drawing
def recognize_drawing(editor_data, auto_center, enhance_strokes):
    if editor_data is None or editor_data.get("composite") is None:
        raise gr.Error("Please draw something first.")

    image = load_composite_image(editor_data)
    processed = preprocess_image(image, crop_to_drawing=auto_center, enhance_strokes=enhance_strokes,)
    if processed.size != input_size: raise gr.Error("Unexpected image size.")
    if np.max(processed) < 0.02: raise gr.Error("The canvas appears blank.")
    category, confidence, probabilities = predict_doodle(processed, params, categories)
    method = "Auto-crop" if auto_center else "Original"

    print(
        f"[{method}] "
        f"Predicted: {category} | "
        f"Confidence: {confidence:.2%}"
    )

    scores = {name: float(probability) for name, probability in zip(categories, probabilities)}
    preview_pixels = np.clip(processed * 255, 0, 255).astype(np.uint8)
    preview = Image.fromarray(preview_pixels)
    preview = preview.resize((280, 280), Image.Resampling.NEAREST)

    print("Original image:", image.size, image.mode)
    print("Processed shape:", processed.shape)
    print("Processed pixel range:", processed.min(), processed.max())
    print("Ink coverage:", f"{np.mean(processed > 0.1):.2%}")

    return scores, preview



# Gradio Blocks interface
with gr.Blocks(title="Doodle Recognizer", fill_width=True) as demo:
    gr.Markdown("# Doodle Recognizer")
    gr.Markdown("Draw a cat, apple, car, house, tree, star, fish, clock, umbrella, or sun.")
    with gr.Row():
        # LEFT: Drawing canvas
        with gr.Column(scale=3, min_width=600):
            drawing = gr.Sketchpad(
                value=Image.new("RGB", (850, 850), "white"),
                type="pil",
                canvas_size=(850, 850),
                fixed_canvas=True,
                width="100%",
                height="calc(100dvh - 180px)",
                format="png",
                layers=False,
                label="Draw your doodle",
                brush = gr.Brush(default_size=20, colors=["#000000"], color_mode="fixed"),
            )

            gr.Markdown("Tip: Medium-thick strokes generally work better for this version of the model.")

        # RIGHT: Predictions, preview, and controls
        with gr.Column(scale=1, min_width=320):
            predictions = gr.Label(
                num_top_classes=3,
                label="Model Predictions"
            )

            processed_preview = gr.Image(
                label="What the model sees (28x28 enlarged)",
                type="pil",
                interactive=False,
                height=280
            )

            auto_center = gr.Checkbox(
                label="Auto-crop and center drawing",
                value=True
            )

            enhance_strokes = gr.Checkbox(
                label="Enhance thin strokes (experimental)",
                value=False
            )

            recognize_button = gr.Button(
                "Recognize Drawing",
                variant="primary"
            )

    recognize_button.click(
        fn=recognize_drawing,
        inputs=[drawing, auto_center, enhance_strokes],
        outputs=[predictions, processed_preview],
        preprocess=False
    )

if __name__ == "__main__":
    demo.launch()


