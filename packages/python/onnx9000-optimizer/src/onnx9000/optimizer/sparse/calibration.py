"""Module for sparse calibration and loss simulation."""

import math
from typing import Any, Union


class DataLoader:
    """Lightweight DataLoader for calibration datasets."""

    def __init__(self, data: Union[list[dict[str, Any]], str]) -> None:
        """Initialize with data list or path to JSON file.

        Args:
            data: A list of data dictionaries or a path to a JSON file.

        """
        if isinstance(data, str):
            import json

            with open(data) as f:
                self.data = json.load(f)
        else:
            self.data = data
        self.current_idx = 0

    def __iter__(self):
        """Return the iterator object itself.

        Returns:
            The DataLoader instance.

        """
        return self

    def __next__(self) -> dict[str, Any]:
        """Return the next item from the dataset.

        Returns:
            The next data dictionary.

        Raises:
            StopIteration: If there are no more items.

        """
        if self.current_idx >= len(self.data):
            self.current_idx = 0
            raise StopIteration
        item = self.data[self.current_idx]
        self.current_idx += 1
        return item


def cross_entropy_loss(y_pred: list[list[float]], y_true: list[int]) -> float:
    """Calculate Cross-Entropy loss natively.

    Args:
        y_pred: Predicted logits, batch_size x num_classes.
        y_true: True class indices, length batch_size.

    Returns:
        The average cross-entropy loss.

    """
    # y_pred: batch_size x num_classes (logits)
    # y_true: batch_size (class indices)

    loss = 0.0
    batch_size = len(y_pred)

    for b in range(batch_size):
        logits = y_pred[b]
        target = y_true[b]

        # Softmax
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        sum_exps = sum(exps)
        prob = exps[target] / sum_exps

        loss += -math.log(max(prob, 1e-12))

    return loss / batch_size


def evaluate_accuracy(y_pred: list[list[float]], y_true: list[int]) -> float:
    """Calculate accuracy natively.

    Args:
        y_pred: Predicted logits, batch_size x num_classes.
        y_true: True class indices, length batch_size.

    Returns:
        The accuracy as a float between 0 and 1.

    """
    correct = 0
    batch_size = len(y_pred)

    for b in range(batch_size):
        logits = y_pred[b]
        target = y_true[b]
        pred_class = logits.index(max(logits))
        if pred_class == target:
            correct += 1

    return correct / batch_size
