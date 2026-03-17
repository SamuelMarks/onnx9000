import pytest
from unittest.mock import patch, MagicMock
from onnx9000.backends.memory.dlpack import (
    DLDataType,
    DLDevice,
    DLTensor,
    DLManagedTensor,
    from_dlpack,
    from_numpy,
)
import numpy as np
import ctypes


def test_from_numpy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ptr, shape, strides, typestr = from_numpy(arr)

    assert shape == (2, 2)
    assert strides is None
    assert typestr == "<f4" or typestr == ">f4"
    assert ptr.value == arr.ctypes.data


def test_from_numpy_strided():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    # create non-contiguous view
    arr_t = arr.T
    with pytest.raises(ValueError, match="not C_CONTIGUOUS"):
        from_numpy(arr_t)


def test_from_numpy_unsupported():
    with pytest.raises(TypeError, match="Object does not support __array_interface__"):
        from_numpy([1, 2, 3])


def test_from_dlpack_unsupported():
    with pytest.raises(TypeError, match="Object does not support DLPack protocol"):
        from_dlpack([1, 2, 3])


class MockDLPackTensor:
    def __init__(self, arr):
        self.arr = arr

    def __dlpack__(self):
        # We need to construct a real PyCapsule to test from_dlpack
        # But constructing a PyCapsule from python is tricky without C extension
        # We will mock the ctypes.pythonapi
        return "capsule"


def test_from_dlpack_mock():
    t = MockDLPackTensor(np.array([1.0, 2.0], dtype=np.float32))

    with patch("ctypes.pythonapi.PyCapsule_GetPointer") as mock_get_ptr:
        mock_get_ptr.return_value = False
        with pytest.raises(RuntimeError, match="Failed to extract DLManagedTensor"):
            from_dlpack(t)

        managed = DLManagedTensor()
        managed.dl_tensor.ndim = 1
        managed.dl_tensor.shape = (ctypes.c_int64 * 1)(2)
        managed.dl_tensor.strides = (ctypes.c_int64 * 1)(1)
        managed.dl_tensor.data = 123
        managed.dl_tensor.device.device_type = 1

        ptr = ctypes.pointer(managed)
        mock_get_ptr.return_value = ptr

        data_ptr, shape, strides, dtype, dev = from_dlpack(t)
        assert data_ptr.value == 123
        assert shape == (2,)
        assert strides == (1,)
        assert dev == 1

        # Test non-contiguous
        managed.dl_tensor.strides = (ctypes.c_int64 * 1)(2)
        with pytest.raises(ValueError, match="not C_CONTIGUOUS"):
            from_dlpack(t)


def test_from_numpy_strided_success():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # modify __array_interface__ directly
    class MockArr:
        __array_interface__ = {
            "data": (123, False),
            "shape": (2, 2),
            "typestr": "<f4",
            "strides": (8, 4),
        }

    ptr, shape, strides, typestr = from_numpy(MockArr())
    assert shape == (2, 2)
    assert strides == (8, 4)


def test_from_dlpack_strided_success():
    class MockDLPackTensor:
        def __dlpack__(self):
            return "capsule"

    t = MockDLPackTensor()
    with patch("ctypes.pythonapi.PyCapsule_GetPointer") as mock_get_ptr:
        managed = DLManagedTensor()
        managed.dl_tensor.ndim = 2
        managed.dl_tensor.shape = (ctypes.c_int64 * 2)(2, 2)
        managed.dl_tensor.strides = (ctypes.c_int64 * 2)(2, 1)
        managed.dl_tensor.data = 123
        managed.dl_tensor.device.device_type = 1

        ptr = ctypes.pointer(managed)
        mock_get_ptr.return_value = ptr

        data_ptr, shape, strides, dtype, dev = from_dlpack(t)
        assert data_ptr.value == 123
        assert shape == (2, 2)
        assert strides == (2, 1)
