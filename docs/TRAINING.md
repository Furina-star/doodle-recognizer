# Dataset and Training

## Dataset

The project uses the **full NumPy bitmap** version of the [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset). Each sample contains 784 grayscale pixel values representing a 28 × 28 image.

The 10 supported categories, in model-output order, are:

`cat`, `apple`, `car`, `house`, `tree`, `star`, `fish`, `clock`, `umbrella`, `sun`.

`data_loader.py` reads the first **5,000 records per category**, converts them to `float32`, and divides pixel values by 255. It combines the categories, shuffles them with seed `42`, and creates two stratified splits using `random_state=42`.

| Dataset split | Percentage | Images | Images per class |
|---|---:|---:|---:|
| Training | 70% | 35,000 | 3,500 |
| Validation | 15% | 7,500 | 750 |
| Test | 15% | 7,500 | 750 |
| **Total** | **100%** | **50,000** | **5,000** |

The selected records and their original ordering must remain the same to reproduce the exact split. Raw `.npy` files are intentionally excluded from Git.

## Training configuration

| Setting | Value |
|---|---|
| Network | `784 -> 64 -> 32 -> 10` |
| Hidden activation | ReLU |
| Output activation | Softmax |
| Loss | Categorical cross-entropy |
| Optimizer | Mini-batch gradient descent (SGD without momentum) |
| Batch size | 32 |
| Learning rate | 0.01 |
| Maximum epochs | 30 |
| Early-stopping patience | 5 epochs |
| L2 weight penalty (`l2_lambda`) | 0.001 |
| Weight initialization | He initialization |
| Weight initialization seed | 123 |
| Batch-shuffling seed | 456 |

Weights are initialized as `float32`; biases are initialized to zero.

## Training procedure

`main.py` calls `train_model()` in `train.py`:

1. Load the training, validation, and test splits; only training and validation data are used to fit or select the model.
2. Initialize the network parameters and perform numerical gradient checking. Training stops if the gradient-check results fail the configured tolerance.
3. Shuffle training samples at the start of each epoch and process mini-batches of 32 samples.
4. Calculate predictions and cross-entropy, perform backpropagation, and update weights and biases. L2 regularization is applied to weight gradients, not bias gradients.
5. After each epoch, calculate validation loss and validation accuracy.
6. Save an in-memory copy of the parameters whenever **validation loss decreases**. Stop early after five consecutive epochs without improvement, or at the 30-epoch limit.
7. Use the copied best-loss checkpoint for validation reporting and save it as `model/doodle_ann_v1.npz`.

The model is selected by **lowest validation loss**, not by maximum validation accuracy. The recorded training loss is cross-entropy and does not include the L2 penalty term.

## Gradient checking

`gradient_check()` compares analytical backpropagation gradients with central finite-difference estimates:

```text
numerical_gradient = (L(theta + epsilon) - L(theta - epsilon)) / (2 * epsilon)
```

The checker converts its inputs and parameters to `float64`, samples up to five entries from each parameter array, and uses `epsilon=1e-5` and tolerance `1e-4`. This check helps identify incorrect derivatives before training; it does not replace the regression tests.

## Saved model

`main.py` saves the six parameter arrays (`W1`, `b1`, `W2`, `b2`, `W3`, `b3`) and metadata:

- `categories`: category names in output-neuron order
- `input_size`: 784
- `normalization_divisor`: 255.0

`predict.py` loads this `.npz` file without retraining.

## Running training

Place the 10 category files in `data/raw/`, following the naming pattern `full_numpy_bitmap_cat.npy`. Install the repository dependencies first, then run from the project root:

```bash
python main.py
```

**Warning:** Running `main.py` retrains the network and overwrites `model/doodle_ann_v1.npz` and training plots. Use `python app.py` to run the saved model without retraining.

## Training plots

The training script generates these files from training and validation metrics:

- [Training and validation loss](../plots/loss_curves.png)
- [Training and validation accuracy](../plots/accuracy_curves.png)
- [Validation confusion matrix](../plots/confusion_matrix.png)

Final held-out results are documented in [EVALUATION.md](EVALUATION.md).
