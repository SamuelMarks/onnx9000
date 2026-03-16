"""Module providing core logic and structural definitions."""

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.frontend.frontend.tensor import Tensor


def test_tensor_constructors() -> None:
    """Tests the corresponding tensor functionality."""
    t1 = Tensor(data=[1.0, 2.0, 3.0])
    assert t1.shape == (3,)
    assert t1.dtype == DType.FLOAT64
    t2 = Tensor(data=np.array([[1, 2], [3, 4]], dtype=np.int32))
    assert t2.shape == (2, 2)
    assert t2.dtype == DType.INT32
    t3 = Tensor(data=True)
    assert t3.shape == ()
    assert t3.dtype == DType.BOOL
    t4 = Tensor(data=b"hello")
    assert t4.dtype == DType.FLOAT32
