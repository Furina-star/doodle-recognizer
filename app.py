"""
A simple Gradio app for recognizing doodles.
"""

import gradio as gr
import numpy as np

from PIL import Image
from predict import load_model, predict_doodle
from preprocessing import preprocess_image

# Load the model parameters and categories
params, categories, input_size, divisor = load_model()

# Gradio interface function to recognize the drawing
def recognize_drawing(editor_data, auto_center):
    if editor_data is None or editor_data.get("composite") is None:
        raise gr.Error("Please draw something first.")

    image = editor_data["composite"]

    processed = preprocess_image(image, crop_to_drawing=auto_center)

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
        auto_center = gr.Checkbox(
            label="Auto-crop and center drawing (experimental)",
            value=True
        )

        with gr.Column(scale=3, min_width=480):
            drawing = gr.Sketchpad(
                value=Image.new("RGB", (420, 420), "white"),
                type="pil",
                canvas_size=(420, 420),
                fixed_canvas=True,
                height=530,
                label="Draw your doodle"
            )

        with gr.Column(scale=2, min_width=320):
            predictions = gr.Label(
                num_top_classes=3,
                label="Model Predictions"
            )

            processed_preview = gr.Image(
                label="What the ANN sees (28x28 enlarged)",
                type="pil",
                interactive=False,
                height=280
            )

    recognize_button = gr.Button(
        "Recognize Drawing",
        variant="primary"
    )

    recognize_button.click(
        fn=recognize_drawing,
        inputs=[drawing, auto_center],
        outputs=[predictions, processed_preview]
    )

if __name__ == "__main__":
    demo.launch()


