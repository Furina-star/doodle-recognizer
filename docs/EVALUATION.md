# Model Evaluation

## Evaluation protocol

The dataset consists of 50,000 Quick Draw bitmap images from 10 categories. Stratified splitting assigns 35,000 images to training, 7,500 to validation, and 7,500 to testing.

Validation loss is used to select the saved model checkpoint. The test set is reserved for final evaluation and is not used for gradient updates or checkpoint selection.

`evaluate.py` reports validation metrics during the training workflow and saves the **validation** confusion matrix and learning curves. `evaluate_test.py` independently loads `model/doodle_ann_v1.npz`, reconstructs the deterministic split, and computes final **test** metrics without training.

## Final results

| Metric | Result |
|---|---:|
| Validation accuracy at selected checkpoint | 87.81% |
| Validation cross-entropy loss | 0.4192 |
| Final test accuracy | **87.67%** |
| Final test cross-entropy loss | **0.4262** |
| Correct test predictions | 6,575 / 7,500 |
| Incorrect test predictions | 925 / 7,500 |
| Validation-to-test accuracy difference | 0.14 percentage points |

The small accuracy difference is consistent with similar performance across the two held-out Quick Draw splits. It does not establish accuracy on arbitrary browser drawings or on the complete Quick Draw dataset.

## Test recall by class

Recall is `correct predictions for the class / actual test examples of the class`. Each class has 750 test images.

| Actual category | Correct / Total | Recall |
|---|---:|---:|
| Cat | 625 / 750 | 83.33% |
| Apple | 688 / 750 | 91.73% |
| Car | 647 / 750 | 86.27% |
| House | 672 / 750 | 89.60% |
| Tree | 659 / 750 | 87.87% |
| Star | 604 / 750 | 80.53% |
| Fish | 662 / 750 | 88.27% |
| Clock | 693 / 750 | 92.40% |
| Umbrella | 657 / 750 | 87.60% |
| Sun | 668 / 750 | 89.07% |

**Clock** has the highest recall (92.40%); **Star** has the lowest (80.53%).

## Frequent classification errors

The final test confusion matrix includes these errors:

| Actual category | Predicted category | Images |
|---|---|---:|
| Star | Sun | 55 |
| Star | Cat | 51 |
| Umbrella | Tree | 47 |
| Cat | Star | 40 |
| Car | Cat | 40 |
| Fish | Cat | 39 |

These counts describe the held-out Quick Draw test set. The file [`plots/confusion_matrix.png`](../plots/confusion_matrix.png) is the **validation** confusion matrix, not the final test matrix. The full test matrix is printed by `evaluate_test.py`.

## Run the final evaluation

Download the original 10 `.npy` category files into `data/raw/`. From the project root:

```bash
python evaluate_test.py
```

The command prints test accuracy, cross-entropy loss, per-class recall, and the full 10 × 10 confusion matrix. It does not modify the saved model. This dataset is not included in the Git repository.

To run the separate regression tests without the raw dataset:

```bash
python -m unittest discover -s tests -v
```

These tests verify model loading, inference inputs and probabilities, and preprocessing behavior. They are not a measurement of classification accuracy.

## Interpretation and limitations

- Reported accuracy is measured on the selected Quick Draw bitmap samples, not on the Gradio drawing canvas.
- Differences in user drawing style, line thickness, and positioning may change predictions.
- A high Softmax score is not proof that a prediction is correct or a calibrated probability of correctness.
- Experimental preprocessing should not be evaluated or tuned using the final held-out test set.

See [PREPROCESSING.md](PREPROCESSING.md) for browser-drawing experiments and limitations.
