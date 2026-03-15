"""Module providing core logic and structural definitions."""

from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType
import numpy as np


def test_tensor_constructors():
    """Provides semantic functionality and verification."""
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
