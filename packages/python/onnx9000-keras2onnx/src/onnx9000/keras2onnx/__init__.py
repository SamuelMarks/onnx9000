"""Keras v2/v3/H5 to ONNX converter and ONNX-to-Keras codegen bridge."""

from onnx9000.converters.tf.api import convert_keras_to_onnx
from onnx9000.core.codegen.keras import ONNXToKerasVisitor
from onnx9000.core.exceptions import ConversionError
from onnx9000.core.ir import Graph


class Keras2OnnxConverter:
    """Bidirectional converter between Keras models and ONNX9000 IR."""

    def to_onnx(self, model_data: bytes, is_v3: bool = False) -> Graph:
        """Convert serialized Keras H5 or Keras v3 model bytes to an ONNX9000 `Graph`.

        Args:
            model_data: Serialized Keras model payload.
            is_v3: `True` if `model_data` is a Keras v3 archive/JSON payload.

        Returns:
            Converted ONNX9000 `Graph`.
        """
        return convert_keras_to_onnx(model_data, is_v3=is_v3)

    def to_keras_code(self, graph: Graph) -> str:
        """Generate Keras 3 Python source code from an ONNX9000 `Graph`.

        Args:
            graph: Source ONNX9000 `Graph`.

        Returns:
            Generated Keras 3 `keras.Model` Python source code.
        """
        return ONNXToKerasVisitor(graph).generate()


def convert(model_str: str) -> str:
    """Convert a Keras model descriptor string into an ONNX-IR summary string.

    Args:
        model_str: Non-empty Keras model descriptor or JSON string.

    Returns:
        Formatted ONNX-IR conversion string.

    Raises:
        ValueError: If `model_str` is empty.
    """
    if not model_str:
        raise ConversionError("Invalid model string")
    return f"[ONNX-IR] from keras {model_str}"


__all__ = [
    "Keras2OnnxConverter",
    "ONNXToKerasVisitor",
    "convert",
    "convert_keras_to_onnx",
]
