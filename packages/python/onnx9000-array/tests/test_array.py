"""Tests for packages/python/onnx9000-array/tests/test_array.py."""

import pytest
import onnx9000_array as np


def test_core_instantiation():
    """Test core instantiation."""
    a = np.array([1, 2, 3])
    assert a.dtype == "float32"
    assert a.data == [1, 2, 3]
    assert a.numpy() == [1, 2, 3]
    assert a.ndim == 1
    a.cpu()
    a.gpu()
    a.quantize_dynamic()


def test_lazy_context():
    """Test lazy context."""
    np.lazy_mode(True)
    x = np.Input("x", [1, 2], "float32")
    assert isinstance(x, np.LazyTensor)
    assert x.op_type == "Input"
    y = np.add(x, 2)
    assert isinstance(y, np.LazyTensor)
    assert y.op_type == "Add"
    z = np.matmul(x, y)
    assert isinstance(z, np.LazyTensor)
    assert z.op_type == "MatMul"
    np.lazy_mode(False)


def test_math_operations_eager():
    """Test math operations eager."""
    a = np.array([1, 2])
    b = np.add(a, 2)
    assert isinstance(b, np.EagerTensor)
    assert isinstance(np.sin(a), np.EagerTensor)
    assert isinstance(np.exp(a), np.EagerTensor)
    assert isinstance(np.reshape(a, [2, 1]), np.EagerTensor)


def test_nn_operations():
    """Test nn operations."""
    x = np.array([1, 2])
    np.lazy_mode(True)
    z = np.nn.relu(x)
    assert isinstance(z, np.LazyTensor)
    assert z.op_type == "Relu"
    np.lazy_mode(False)


def test_linalg_operations():
    """Test linalg operations."""
    x = np.array([1, 2])
    np.lazy_mode(True)
    z = np.linalg.det(x)
    assert isinstance(z, np.LazyTensor)
    assert z.op_type == "Det"
    np.lazy_mode(False)
