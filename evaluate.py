"""
This module contains functions for evaluating the Doodle Recognizer model.
It includes generating confusion matrices and plotting training/validation loss and accuracy curves.
"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.ticker import PercentFormatter

# Plotting and evaluation functions for the Doodle Recognizer model
def evaluate_model(y_val, y_pred, categories, history, best_val_acc, plots_directory="plots"):
    num_classes = len(categories)

    plots_directory = Path(plots_directory)
    plots_directory.mkdir(parents=True, exist_ok=True)

    epoch_completed = len(history["train_loss"])
    epoch_numbers = range(1, epoch_completed + 1)

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)

    print("\nConfusion Matrix:")
    for actual, predicted in zip(y_val, y_pred):
        cm[actual, predicted] += 1
    print(cm)

    correct_per_class = np.diag(cm)
    total_per_class = cm.sum(axis=1)
    recall = correct_per_class / total_per_class

    assert cm.sum() == len(y_val)
    cm_accuracy = np.trace(cm) / cm.sum()
    assert np.isclose(cm_accuracy, best_val_acc)

    class_names = [category.title() for category in categories]

    for name, value in zip(class_names, recall):
        print(f"{name} Recall: {value:.2%}")
    print("Total predictions:", cm.sum(), "\n")

    # Plot the confusion matrix
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Doodle Recognizer - Confusion Matrix")
    display.figure_.savefig(plots_directory / "confusion_matrix.png", dpi=200, bbox_inches="tight")

    # Plot the training and validation loss
    plt.figure(figsize=(8, 5))

    plt.plot(epoch_numbers, history["train_loss"], label="Training Loss")
    plt.plot(epoch_numbers, history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Doodle Recognizer - Loss Curves")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_directory / "loss_curves.png", dpi=200, bbox_inches="tight")

    # Plot the training and validation accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(epoch_numbers, history["train_acc"], label="Training Accuracy")
    plt.plot(epoch_numbers, history["val_acc"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Doodle Recognizer - Accuracy Curves")

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xticks(epoch_numbers)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_directory / "accuracy_curves.png", dpi=200, bbox_inches="tight")

    # Print the saved plot files
    for file in sorted(plots_directory.glob("*.png")):
        print(f"Saved plot: {file}")

    return cm, recall, cm_accuracy
