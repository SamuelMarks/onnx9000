"""Structs."""

import ctypes


class Dims(ctypes.Structure):
    """Dims class."""

    _fields_ = [("nbDims", ctypes.c_int32), ("d", ctypes.c_int32 * 8)]

    def __init__(self, shape: list[int]):
        """Initialize."""
        super().__init__()
        self.nbDims = len(shape)
        if self.nbDims > 8:
            raise ValueError("TensorRT dimensions cannot exceed 8")
        for i, s in enumerate(shape):
            self.d[i] = s


class Weights(ctypes.Structure):
    """Weights class."""

    _fields_ = [
        ("type", ctypes.c_int32),  # DataType
        ("values", ctypes.c_void_p),
        ("count", ctypes.c_int64),
    ]
