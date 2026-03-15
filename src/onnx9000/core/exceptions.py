"""Custom exception definitions for the ONNX9000 compiler and runtime."""


class Onnx9000Error(Exception):
    """Base class for exceptions in this module."""

    pass


class CompilationError(Onnx9000Error):
    """Exception raised for errors during JIT compilation (e.g. g++ / emcc failure)."""

    pass


class UnsupportedOpError(Onnx9000Error):
    """Exception raised when attempting to compile an unsupported ONNX operator."""

    def __init__(self, op_type: str, message: str = ""):
        """Initialize UnsupportedOpError with the failing operator type and an optional message."""

        self.op_type = op_type
        self.message = message or f"Operator '{op_type}' is not supported yet."
        super().__init__(self.message)


class ShapeMismatchError(Onnx9000Error):
    """Exception raised when tensor shapes do not match requirements for an operation."""

    pass
