"""Comprehensive unit tests for `onnx9000.array` and `onnx9000_array`."""

import onnx9000.array as arr
import onnx9000_array as np
import pytest


def test_core_instantiation_and_indexing() -> None:
    """Verify EagerTensor instantiation, properties, real __getitem__, and real __setitem__."""
    np.lazy_mode(False)
    a = np.array([10.0, 20.0, 30.0])
    assert a.dtype == "float32"
    assert a.data == [10.0, 20.0, 30.0]
    assert a.numpy() == [10.0, 20.0, 30.0]
    assert a.ndim == 1
    assert a.cpu() is a
    assert a.gpu() is a
    assert a.quantize_dynamic() is a
    assert a.evaluate() is a

    # Integer indexing and slice indexing
    elem_1 = a[1]
    assert elem_1.data == [20.0]
    slice_0_2 = a[0:2]
    assert slice_0_2.data == [10.0, 20.0]

    # __setitem__ by index and by slice
    a[1] = 25.0
    assert a.data == [10.0, 25.0, 30.0]
    a[0:2] = [1.0, 2.0]
    assert a.data == [1.0, 2.0, 30.0]
    a[0] = [9.0, 8.0]
    assert a.data == [9.0, 8.0]

    # Non-list data_ref branches for __getitem__ and __setitem__
    b = np.EagerTensor(b"\x01\x02")
    assert b[0].data == [1.0]
    b[0] = 5.0
    assert b.data == [5.0]
    b.data = [7.0]
    assert b.data == [7.0]


def test_eager_tensor_methods_and_operators() -> None:
    """Verify EagerTensor arithmetic methods and dunder operators."""
    np.lazy_mode(False)
    a = np.array([2.0, 4.0])
    assert isinstance(a.T, np.EagerTensor)
    assert isinstance(a.reshape((2, 1)), np.EagerTensor)
    assert isinstance(a.add(1.0), np.EagerTensor)
    assert isinstance(a + 1.0, np.EagerTensor)
    assert isinstance(a.subtract(1.0), np.EagerTensor)
    assert isinstance(a.__subtract__(1.0), np.EagerTensor)
    assert isinstance(a - 1.0, np.EagerTensor)
    assert isinstance(a.multiply(2.0), np.EagerTensor)
    assert isinstance(a.__multiply__(2.0), np.EagerTensor)
    assert isinstance(a * 2.0, np.EagerTensor)
    assert isinstance(a.divide(2.0), np.EagerTensor)
    assert isinstance(a.__divide__(2.0), np.EagerTensor)
    assert isinstance(a / 2.0, np.EagerTensor)
    assert isinstance(a.power(2.0), np.EagerTensor)
    assert isinstance(a.__power__(2.0), np.EagerTensor)
    assert isinstance(a**2.0, np.EagerTensor)
    assert isinstance(a.mod(2.0), np.EagerTensor)
    assert isinstance(a % 2.0, np.EagerTensor)


def test_lazy_context_and_errors() -> None:
    """Verify LazyTensor tracing, symbolic Input dimensions, and exception classes."""
    np.lazy_mode(True)
    x = np.Input("x", [1, "batch"], "float32")
    assert isinstance(x, np.LazyTensor)
    assert x.op_type == "Input"
    assert x.name == "x"
    y = np.add(x, 2)
    assert isinstance(y, np.LazyTensor)
    assert y.op_type == "Add"
    z = np.matmul(x, y)
    assert isinstance(z, np.LazyTensor)
    assert z.op_type == "MatMul"
    np.lazy_mode(False)

    with pytest.raises(arr.BroadcastError, match="Cannot broadcast shapes"):
        raise np.BroadcastError((1, 2), (3, 4))

    with pytest.raises(arr.TypeMismatchError, match="Type mismatch"):
        raise np.TypeMismatchError("float32", "int32")

    def fn(v: np.EagerTensor) -> np.EagerTensor:
        """Execute fn operation.

        Args:
            v: Parameter specifying v (np.EagerTensor).

        Returns:
            Result of fn (np.EagerTensor).
        """
        return v

    assert np.onnx_function(fn) is fn
