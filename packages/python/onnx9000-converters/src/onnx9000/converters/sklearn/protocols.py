"""Structural `Protocol` definitions for Scikit-Learn estimators and transformers.

Replaces untyped `estimator: object` annotations across `onnx9000.converters.sklearn`
with explicit structural typing protocols.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class FloatArrayProtocol(Protocol):
    """Protocol representing a NumPy-like numerical array with `flatten()` and `tolist()`."""

    @property
    def size(self) -> int:
        """Return total number of elements in the array."""

    def flatten(self) -> "FloatArrayProtocol":
        """Return a flattened 1D view or copy of the array."""

    def tolist(self) -> list[float]:
        """Convert the array elements to a Python list of floats."""

    def __neg__(self) -> "FloatArrayProtocol":
        """Negate array elements."""

    def __truediv__(self, other: "FloatArrayProtocol | float") -> "FloatArrayProtocol":
        """Divide array elements."""

    def __rtruediv__(self, other: float) -> "FloatArrayProtocol":
        """Right-divide scalar by array elements."""

    def __len__(self) -> int:
        """Return length of the first dimension."""


@runtime_checkable
class SKLearnEstimatorProtocol(Protocol):
    """Base protocol for Scikit-Learn estimators, classifiers, regressors, and transformers."""

    pass


@runtime_checkable
class LinearEstimatorProtocol(SKLearnEstimatorProtocol, Protocol):
    """Protocol for Scikit-Learn linear regressors and classifiers."""

    coef_: FloatArrayProtocol
    intercept_: FloatArrayProtocol
    classes_: Sequence[int | str | bytes]


@runtime_checkable
class ScalerEstimatorProtocol(SKLearnEstimatorProtocol, Protocol):
    """Protocol for Scikit-Learn preprocessing scalers, normalizers, and encoders."""

    mean_: FloatArrayProtocol
    scale_: FloatArrayProtocol
    min_: FloatArrayProtocol
    center_: FloatArrayProtocol
    norm: str
    threshold: float
    categories_: Sequence[Sequence[int | str | bytes]]
    classes_: Sequence[int | str | bytes]


@runtime_checkable
class TreeEstimatorProtocol(SKLearnEstimatorProtocol, Protocol):
    """Protocol for Scikit-Learn decision tree and ensemble estimators."""

    classes_: Sequence[int | str | bytes]


@runtime_checkable
class SVMEstimatorProtocol(SKLearnEstimatorProtocol, Protocol):
    """Protocol for Scikit-Learn Support Vector Machine classifiers and regressors."""

    kernel: str
    support_vectors_: FloatArrayProtocol
    dual_coef_: FloatArrayProtocol
    intercept_: FloatArrayProtocol
    probA_: FloatArrayProtocol
    probB_: FloatArrayProtocol
    classes_: Sequence[int | str | bytes]


@runtime_checkable
class NaiveBayesEstimatorProtocol(SKLearnEstimatorProtocol, Protocol):
    """Protocol for Scikit-Learn Naive Bayes and Nearest Neighbors estimators."""


@runtime_checkable
class ClusterEstimatorProtocol(SKLearnEstimatorProtocol, Protocol):
    """Protocol for Scikit-Learn clustering and mixture estimators."""


@runtime_checkable
class DecompositionEstimatorProtocol(SKLearnEstimatorProtocol, Protocol):
    """Protocol for Scikit-Learn matrix decomposition and manifold estimators."""


@runtime_checkable
class CompositeEstimatorProtocol(SKLearnEstimatorProtocol, Protocol):
    """Protocol for Scikit-Learn meta-estimators (`Pipeline`, `GridSearchCV`, `ColumnTransformer`)."""

    best_estimator_: "CompositeEstimatorProtocol"
    steps: Sequence[tuple[str, "CompositeEstimatorProtocol | str"]]
    transformer_list: Sequence[tuple[str, "CompositeEstimatorProtocol | str"]]
    transformers_: Sequence[tuple[str, "CompositeEstimatorProtocol | str", Sequence[int | str]]]
    remainder: str
    mean_: FloatArrayProtocol
    scale_: FloatArrayProtocol
    coef_: FloatArrayProtocol
    intercept_: FloatArrayProtocol
    classes_: Sequence[int | str | bytes]


__all__ = [
    "ClusterEstimatorProtocol",
    "CompositeEstimatorProtocol",
    "DecompositionEstimatorProtocol",
    "FloatArrayProtocol",
    "LinearEstimatorProtocol",
    "NaiveBayesEstimatorProtocol",
    "SKLearnEstimatorProtocol",
    "SVMEstimatorProtocol",
    "ScalerEstimatorProtocol",
    "TreeEstimatorProtocol",
]
