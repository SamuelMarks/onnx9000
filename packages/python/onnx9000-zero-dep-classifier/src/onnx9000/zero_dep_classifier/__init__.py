"""Pure-Python zero-dependency image preprocessing and top-k classification pipeline."""

import math
from dataclasses import dataclass

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ClassificationPrediction:
    """Single top-k classification result entry.

    Attributes:
        class_index: Zero-based class index.
        label: Human-readable class label.
        probability: Normalized softmax probability in `[0.0, 1.0]`.
    """

    class_index: int
    label: str
    probability: float


def normalize_rgb_pixel(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Normalize an 8-bit RGB pixel `(0..255)` using standard ImageNet mean and std.

    Args:
        r: Red channel byte `[0, 255]`.
        g: Green channel byte `[0, 255]`.
        b: Blue channel byte `[0, 255]`.

    Returns:
        Normalized `(r_norm, g_norm, b_norm)` float tuple.
    """
    return (
        (r / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
        (g / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
        (b / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
    )


def softmax_probabilities(logits: list[float]) -> list[float]:
    """Compute numerically stable softmax probabilities in pure Python.

    Args:
        logits: Raw unnormalized model output scores.

    Returns:
        Probability distribution summing to `1.0` (or `[]` if `logits` is empty).
    """
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [val / total for val in exps]


class ZeroDepClassifier:
    """Zero-dependency pure-Python image classification pipeline."""

    def __init__(self, labels: list[str] | None = None) -> None:
        """Initialize the classifier with an optional list of class labels.

        Args:
            labels: Optional class label names indexed by class ID.
        """
        self.labels: list[str] = labels if labels is not None else []

    def top_k(self, logits: list[float], k: int = 5) -> list[ClassificationPrediction]:
        """Compute the top-`k` predictions from raw `logits`.

        Args:
            logits: Raw unnormalized logit scores.
            k: Number of highest-probability classes to return.

        Returns:
            List of `ClassificationPrediction` sorted by descending probability.
        """
        probs = softmax_probabilities(logits)
        indexed = sorted(enumerate(probs), key=lambda pair: pair[1], reverse=True)
        results: list[ClassificationPrediction] = []
        for idx, prob in indexed[: max(0, k)]:
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            results.append(ClassificationPrediction(class_index=idx, label=label, probability=prob))
        return results

    def process(self, arg: str) -> str:
        """Process an image or classifier target identifier.

        Args:
            arg: Target identifier string.

        Returns:
            Formatted Zero Dep Classifier status string.
        """
        return f"Zero Dep Classifier processed {arg}"


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ClassificationPrediction",
    "ZeroDepClassifier",
    "normalize_rgb_pixel",
    "softmax_probabilities",
]
