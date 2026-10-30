"""Unified custom exception hierarchy for the ONNX9000 compiler, converters, optimizer, and runtime."""


class Onnx9000Error(ValueError, RuntimeError, TypeError):
    """Base class for all exceptions in the ONNX9000 ecosystem."""

    def __init__(self, message: str = "An error occurred") -> None:
        """Initialize the base Onnx9000Error.

        Args:
            message: Human-readable description of the error.
        """
        self.message: str = str(message)
        super().__init__(self.message)


class CompilationError(Onnx9000Error):
    """Exception raised for errors during AOT or JIT compilation."""


class UnsupportedOpError(Onnx9000Error):
    """Exception raised when encountering an unsupported ONNX or frontend operator."""

    def __init__(self, op_type: str, message: str = "") -> None:
        """Initialize UnsupportedOpError with the failing operator type and an optional message.

        Args:
            op_type: Name of the unsupported operator.
            message: Optional detailed error message override.
        """
        self.op_type: str = op_type
        resolved_message = message or f"Operator '{op_type}' is not supported yet."
        super().__init__(resolved_message)


class ShapeMismatchError(Onnx9000Error):
    """Exception raised when tensor shapes do not satisfy operator constraints."""


class ONNXParseError(Onnx9000Error):
    """Exception raised when failing to parse an ONNX Protobuf or model buffer."""


class ShapeInferenceError(Onnx9000Error):
    """Exception raised when static or symbolic shape inference fails."""


class UnsupportedOpsetError(Onnx9000Error):
    """Exception raised when validating or lowering a model with an unsupported opset version."""


class ValidationError(Onnx9000Error):
    """Exception raised when a model fails static ONNX checker validation rules."""


class ConversionError(Onnx9000Error):
    """Exception raised when converting a source framework model into ONNX9000 IR fails."""


class OptimizerError(Onnx9000Error):
    """Exception raised when a graph optimization, fusion, or pruning pass fails."""


class AutogradError(Onnx9000Error):
    """Exception raised when backward graph VJP construction or AOT autograd fails."""


class BackendExecutionError(Onnx9000Error):
    """Exception raised when runtime execution on a hardware backend fails."""


class SafetensorsError(Onnx9000Error):
    """Exception raised when parsing, validating, or serializing a Safetensors file fails."""


class GGUFError(Onnx9000Error):
    """Exception raised when reading or writing a GGUF quantized model container fails."""


class TokenizationError(Onnx9000Error):
    """Exception raised when tokenizer vocabulary loading, encoding, or decoding fails."""


class CLIError(Onnx9000Error):
    """Exception raised when CLI command arguments or execution fail."""


__all__ = [
    "AutogradError",
    "BackendExecutionError",
    "CLIError",
    "CompilationError",
    "ConversionError",
    "GGUFError",
    "ONNXParseError",
    "Onnx9000Error",
    "OptimizerError",
    "SafetensorsError",
    "ShapeInferenceError",
    "ShapeMismatchError",
    "TokenizationError",
    "UnsupportedOpError",
    "UnsupportedOpsetError",
    "ValidationError",
]
