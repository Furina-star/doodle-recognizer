"""
This script evaluates the performance of a trained neural network model on a test dataset.
It loads the model parameters, test data, and computes the accuracy, loss, and confusion matrix
for the predictions made by the model. The results are printed to the console.
"""

import numpy as np

from sklearn.metrics import confusion_matrix
from ann import forward, cross_entropy
from data_loader import load_datasets
from predict import load_model

params, categories, input_size, divisor = load_model()

_, _, X_test, _, _, y_test = load_datasets(
    categories=categories,
    sample_per_class=5000,
    data_directory="data/raw"
)

assert X_test.shape == (7500, input_size)
assert len(y_test) == 7500

P_test, _ = forward(X_test, *params)
y_pred = np.argmax(P_test, axis=1)

test_accuracy = np.mean(y_pred == y_test)
test_loss = cross_entropy(P_test, y_test)

print("\nFINAL TEST EVALUATION")
print("-" * 45)
print(f"Test samples: {len(y_test)}")
print(f"Test accuracy: {test_accuracy:.2%}")
print(f"Test loss: {test_loss:.4f}")

cm = confusion_matrix(y_test,y_pred, labels=np.arange(len(categories)))

print("\nPER-CLASS PERFORMANCE")
print("-" * 45)

for i, category in enumerate(categories):
    correct = cm[i, i]
    total = cm[i].sum()
    recall = correct / total

    print(
        f"{category:10s} | "
        f"Correct: {correct:3d}/{total:3d} | "
        f"Recall: {recall:.2%}"
    )

print("\nCONFUSION MATRIX")
print(cm)

print("\nFinal evaluation completed.")
