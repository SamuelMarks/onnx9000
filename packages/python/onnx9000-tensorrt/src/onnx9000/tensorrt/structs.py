"""Structs."""

import ctypes

from onnx9000.core.exceptions import BackendExecutionError


class Dims(ctypes.Structure):
    """Dims class."""

    _fields_ = [("nbDims", ctypes.c_int32), ("d", ctypes.c_int32 * 8)]

    def __init__(self, shape: list[int]) -> None:
        """Initialize Dims with configured parameters and state.

        Args:
            shape: Parameter specifying shape (list[int]).

        Raises:
            ValueError: If an invalid state or argument is encountered during execution.
        """
        super().__init__()
        self.nbDims = len(shape)
        if self.nbDims > 8:
            raise BackendExecutionError("TensorRT dimensions cannot exceed 8")
        for i, s in enumerate(shape):
            self.d[i] = s


class Weights(ctypes.Structure):
    """Weights class."""

    _fields_ = [
        ("type", ctypes.c_int32),  # DataType
        ("values", ctypes.c_void_p),
        ("count", ctypes.c_int64),
    ]
