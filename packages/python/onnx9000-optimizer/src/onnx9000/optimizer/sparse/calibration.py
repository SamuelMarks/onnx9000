"""Module for sparse calibration and loss simulation."""

import math
from typing import Any, Dict, List, Optional, Union, Tuple


class DataLoader:
    """Lightweight DataLoader for calibration datasets."""

    def __init__(self, data: Union[List[Dict[str, Any]], str]) -> None:
        """Initialize with data list or path to JSON file."""
        if isinstance(data, str):
            import json

            with open(data, "r") as f:
                self.data = json.load(f)
        else:
            self.data = data
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        if self.current_idx >= len(self.data):
            self.current_idx = 0
            raise StopIteration
        item = self.data[self.current_idx]
        self.current_idx += 1
        return item


def cross_entropy_loss(y_pred: List[List[float]], y_true: List[int]) -> float:
    """Calculate Cross-Entropy loss natively."""
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


def evaluate_accuracy(y_pred: List[List[float]], y_true: List[int]) -> float:
    """Calculate accuracy natively."""
    correct = 0
    batch_size = len(y_pred)

    for b in range(batch_size):
        logits = y_pred[b]
        target = y_true[b]
        pred_class = logits.index(max(logits))
        if pred_class == target:
            correct += 1

    return correct / batch_size
