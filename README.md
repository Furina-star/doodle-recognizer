# Doodle Recognizer

A handwritten doodle classification system built using a feedforward Artificial Neural Network (ANN) implemented from scratch with NumPy. The project uses Google's Quick, Draw! dataset for training and Gradio for its web interface.

The model classifies drawings into 10 categories: cat, apple, car, house, tree, star, fish, clock, umbrella, and sun.

## Features

- ANN implemented using NumPy without TensorFlow or PyTorch.
- Forward propagation and backpropagation.
- ReLU activation in hidden layers and Softmax in the output layer.
- Mini-batch gradient descent with L2 regularization and early stopping.
- Gradient checking to verify backpropagation.
- Gradio drawing interface with predictions and a 28 × 28 image preview.
- Optional image cropping and experimental stroke thickening.
- Saved model inference without retraining.

## Neural Network Architecture

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 784 | None |
| Hidden Layer 1 | 64 | ReLU |
| Hidden Layer 2 | 32 | ReLU |
| Output | 10 | Softmax |

Input images are normalized to values between 0 and 1. The model uses categorical cross-entropy loss and mini-batch gradient descent for training.

## Dataset

The project uses the 28 × 28 grayscale bitmap version of the Google Quick, Draw! dataset.

- Categories: 10
- Samples per category: 5,000
- Total samples: 50,000
- Training set: 35,000 images (70%)
- Validation set: 7,500 images (15%)
- Test set: 7,500 images (15%)

The dataset is shuffled and divided using reproducible, stratified splits with a fixed random seed.

Raw dataset files are excluded from the repository.

Dataset source: [Google Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)

## Model Performance

| Metric | Result |
|---|---:|
| Best validation accuracy | 87.81% |
| Final test accuracy | 87.67% |
| Final test cross-entropy loss | 0.4262 |
| Test samples | 7,500 |

The results were obtained using the selected Quick Draw bitmap dataset. Performance on drawings created through the Gradio interface may differ because of drawing style, stroke thickness, and preprocessing.

## Installation

**1. Clone the repository**

`git clone https://github.com/Furina-star/doodle-recognizer.git`

`cd doodle-recognizer`

**2. Create and activate a virtual environment**

Windows:

`python -m venv .venv`

`.venv\Scripts\activate`

**3. Install dependencies**

`python -m pip install -r requirements.txt`

## Run the Application

Execute:

`python app.py`

Open the local Gradio URL displayed in the terminal.

The trained model is included in `model/doodle_ann_v1.npz`. Downloading the training dataset is not required to run the application.

## Training and Evaluation

To reproduce training, download the full NumPy bitmap `.npy` files for all 10 categories and place them in `data/raw/`.

Expected filename format: `full_numpy_bitmap_cat.npy`, `full_numpy_bitmap_apple.npy`, etc.

**Train the ANN:**

`python main.py`

This runs training, selects the best validation checkpoint, saves the model, and generates evaluation plots. Running it will overwrite the existing saved model.

**Evaluate the saved model:**

`python evaluate_test.py`

The evaluation script requires the raw dataset files.

**Run regression tests:**

`python -m unittest discover -s tests -v`

The regression tests check model loading, prediction outputs, input validation, and image preprocessing without retraining the model.

## Project Structure

| File or Directory | Purpose |
|---|---|
| `ann.py` | Forward propagation, Softmax, cross-entropy, and backpropagation |
| `train.py` | Parameter initialization, gradient checking, and training |
| `data_loader.py` | Dataset loading, normalization, and splitting |
| `main.py` | Training pipeline and model saving |
| `evaluate.py` | Validation metrics and training plots |
| `evaluate_test.py` | Final test-set evaluation |
| `predict.py` | Saved-model loading and prediction |
| `preprocessing.py` | Drawing preprocessing |
| `app.py` | Gradio interface |
| `tests/` | Automated regression tests |
| `model/` | Saved ANN parameters |
| `plots/` | Training and validation visualizations |
| `screenshots/` | Application screenshots |

## Application Preview

[View Gradio interface screenshot](screenshots/whole-gradio-layout.png)

## Limitations

- Classification is limited to the 10 trained categories.
- Predictions depend on drawing style and stroke thickness.
- Softmax confidence is not a guarantee of correctness.
- Experimental stroke thickening can improve some drawings while reducing accuracy on others.

## Technical Documentation

Additional documentation is available in the `docs/` directory:

- [ARCHITECTURE.md](docs/ARCHITECTURE.md): Neural network architecture and mathematical operations.
- [TRAINING.md](docs/TRAINING.md): Dataset preparation and training configuration.
- [EVALUATION.md](docs/EVALUATION.md): Model evaluation and experimental results.
- [PREPROCESSING.md](docs/PREPROCESSING.md): Image processing and interface-related limitations.