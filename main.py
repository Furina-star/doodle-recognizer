import numpy as np
from pathlib import Path
from ann import forward, cross_entropy
from data_loader import load_datasets
from train import gradient_check, initialize_parameters, train_model
from evaluate import evaluate_model

# Define the categories for the doodle recognition task and the number of samples per class
categories = [
    "cat", "apple", "car",
    "house", "tree", "star",
    "fish", "clock", "umbrella",
    "sun"
]
sample_per_class = 5000
num_classes = len(categories)

# Load the datasets (training, validation, and testing) from data_loader.py
X_train, X_val, X_test, y_train, y_val, y_test = load_datasets(
    categories=categories,
    sample_per_class=sample_per_class,
    data_directory="data/raw"
)

# Set hyperparameters for training
batch_size = 32 # Batch size for training
epochs = 30 # Number of epochs to train the model
patience = 5 # Number of epochs to wait for improvement before early stopping
learning_rate = 0.01 # Learning rate for gradient descent
l2_lambda = 0.001 # L2 regularization strength
hidden1 = 64 # Number of neurons in the first hidden layer
hidden2 = 32 # Number of neurons in the second hidden layer

weight_seed = 123 # Seed for weight initialization to ensure reproducibility
batch_seed = 456 # Seed for batch shuffling to ensure reproducibility

rng_weights = np.random.default_rng(weight_seed)
rng_batches = np.random.default_rng(batch_seed)

# Initialize the weights and biases from train.py
input_size = X_train.shape[1]
W1, b1, W2, b2, W3, b3 = initialize_parameters(input_size,hidden1,hidden2,num_classes,rng_weights)

# Gradient Checking (the call is what makes it print)
gradient_tolerance = 1e-4
gradient_errors = gradient_check([W1, b1, W2, b2, W3, b3], X_train[:8], y_train[:8], tol=gradient_tolerance)

if not gradient_errors or any(not np.isfinite(error) or error >= gradient_tolerance for error in gradient_errors.values()):
    raise RuntimeError("Gradient checking failed. Training aborted.")

print("All gradient checks passed. Proceeding to training.")

# Call the train_model function to train the model and obtain the best parameters and training history
best_params, history, best_epoch, best_val_loss, best_val_acc = train_model(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    params=[W1, b1, W2, b2, W3, b3],
    rng_batches=rng_batches,
    batch_size=batch_size,
    epochs=epochs,
    patience=patience,
    learning_rate=learning_rate,
    l2_lambda=l2_lambda
)

# Verify that the best parameters yield the same validation loss as recorded
P_best, _ = forward(X_val, *best_params)
y_pred = np.argmax(P_best, axis=1)
verified_best_loss = cross_entropy(P_best, y_val)
assert np.isclose(verified_best_loss, best_val_loss)  # confirms best_params matches the saved loss

print(f"\nBest Epoch: {best_epoch}")
print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Acc: {best_val_acc:.2%}\n")

# Save the best model parameters to a compressed .npz file
models_path = Path("model/doodle_ann_v1.npz")
models_path.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    models_path,

    W1=best_params[0],
    b1=best_params[1],

    W2=best_params[2],
    b2=best_params[3],

    W3=best_params[4],
    b3=best_params[5],

    categories=np.array(categories),
    input_size=input_size,
    normalization_divisor=255.0
)

print(f"Model saved to: {models_path}")

# Load the saved ANN
with np.load(models_path, allow_pickle=False) as saved_model:
    loaded_params = [
        saved_model["W1"],
        saved_model["b1"],
        saved_model["W2"],
        saved_model["b2"],
        saved_model["W3"],
        saved_model["b3"]
    ]

    loaded_categories = saved_model["categories"].tolist()

    loaded_input_size = int(saved_model["input_size"])
    loaded_divisor = float(saved_model["normalization_divisor"])

print("Loaded categories:", loaded_categories)
print("Loaded input size:", loaded_input_size)
print("Loaded normalization divisor:", loaded_divisor)

parameter_names = ["W1", "b1", "W2", "b2", "W3", "b3"]

for name, parameter in zip(parameter_names, loaded_params):
    print(f"{name}: {parameter.shape}")

# Predictions using loaded parameters
P_loaded, _ = forward(X_val, *loaded_params)

# Compare predictions
difference = np.max(np.abs(P_best - P_loaded))

print("Maximum probability difference:", difference)

assert np.allclose(P_best, P_loaded)
assert np.array_equal(
    np.argmax(P_best, axis=1),
    np.argmax(P_loaded, axis=1)
)

print("Model loading verification: PASSED")

loaded_validation = np.argmax(P_loaded, axis=1)
print(f"Loaded model validation accuracy: {np.mean(loaded_validation == y_val):.2%}")

# Evaluate the model using the evaluate_model function from evaluate.py
cm, recall, evaluation_accuracy = evaluate_model(y_val=y_val, y_pred=y_pred, categories=categories, history=history, best_val_acc=best_val_acc, plots_directory="plots")



