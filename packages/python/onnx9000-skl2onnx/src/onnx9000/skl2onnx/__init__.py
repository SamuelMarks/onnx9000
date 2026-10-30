"""Scikit-Learn and traditional ML (`ai.onnx.ml`) converter for ONNX9000."""

from onnx9000.converters.sklearn import SKLearnParser
from onnx9000.converters.sklearn.protocols import CompositeEstimatorProtocol
from onnx9000.core.exceptions import ConversionError
from onnx9000.core.ir import Graph


def convert_sklearn(model: CompositeEstimatorProtocol, name: str = "sklearn_model") -> Graph:
    """Convert a Scikit-Learn estimator or pipeline into an ONNX9000 `Graph`.

    Args:
        model: Fitted Scikit-Learn estimator satisfying `CompositeEstimatorProtocol`.
        name: Name assigned to the resulting ONNX `Graph`.

    Returns:
        Converted ONNX9000 `Graph`.
    """
    return SKLearnParser(model, name=name).parse()


def convert(model_str: str) -> str:
    """Convert a Scikit-Learn model descriptor string into an ONNX-IR summary string.

    Args:
        model_str: Non-empty Scikit-Learn model descriptor.

    Returns:
        Formatted ONNX-IR conversion string.

    Raises:
        ValueError: If `model_str` is empty.
    """
    if not model_str:
        raise ConversionError("Invalid model string")
    return f"[ONNX-IR] from skl {model_str}"


__all__ = ["SKLearnParser", "convert", "convert_sklearn"]
