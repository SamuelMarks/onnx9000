import pytest
from unittest.mock import patch, MagicMock
import ctypes


def test_apple_bindings():
    import onnx9000.backends.apple.bindings as bindings

    bindings._load_libraries()

    assert isinstance(bindings.is_accelerate_available(), bool)
    assert isinstance(bindings.is_metal_available(), bool)
    assert isinstance(bindings.is_mps_available(), bool)

    old_flib = bindings._foundation_lib
    bindings._foundation_lib = MagicMock()
    bindings._objc = MagicMock()

    with patch("onnx9000.backends.apple.bindings.get_class") as mock_get_class:
        with patch("onnx9000.backends.apple.bindings.get_selector") as mock_get_selector:
            mock_get_class.return_value = ctypes.c_void_p(123)
            mock_get_selector.return_value = ctypes.c_void_p(456)

            with patch("ctypes.cast") as mock_cast:
                mock_msg_send = MagicMock()
                mock_msg_send.return_value = ctypes.c_void_p(789)
                mock_cast.return_value = mock_msg_send

                res = bindings.nsstring("hello")
                assert res.value == 789

    bindings._foundation_lib = None
    res = bindings.nsstring("hello")
    assert res.value is None or res.value == 0

    bindings._foundation_lib = old_flib

    with patch("onnx9000.backends.apple.bindings.is_metal_available", return_value=False):
        assert bindings.mtl_create_system_default_device() is None

    old_metal = bindings._metal_lib
    with patch("onnx9000.backends.apple.bindings.is_metal_available", return_value=True):
        mock_metal = MagicMock()
        mock_func = MagicMock()
        mock_func.return_value = "device"
        mock_metal.MTLCreateSystemDefaultDevice = mock_func
        bindings._metal_lib = mock_metal
        assert bindings.mtl_create_system_default_device() == "device"

        del mock_metal.MTLCreateSystemDefaultDevice
        assert bindings.mtl_create_system_default_device() is None

    bindings._metal_lib = old_metal
    bindings._load_libraries()


def test_get_class_and_selector():
    import onnx9000.backends.apple.bindings as bindings

    old_objc = bindings._objc
    mock_objc = MagicMock()
    mock_objc.objc_getClass.return_value = ctypes.c_void_p(111)
    mock_objc.sel_registerName.return_value = ctypes.c_void_p(222)
    bindings._objc = mock_objc

    c = bindings.get_class("NSString")
    s = bindings.get_selector("stringWithUTF8String:")

    assert c.value == 111
    assert s.value == 222

    bindings._objc = None
    assert bindings.get_class("A").value in (0, None)
    assert bindings.get_selector("B").value in (0, None)

    bindings._objc = old_objc


def test_apple_bindings_import_errors():
    import onnx9000.backends.apple.bindings as bindings

    with patch("ctypes.CDLL", side_effect=Exception("Test Error")):
        with patch("ctypes.util.find_library", side_effect=Exception("Test Error")):
            bindings._load_libraries()
            assert bindings._accelerate_lib is None
            assert bindings._metal_lib is None
            assert bindings._mps_lib is None
            assert bindings._foundation_lib is None
            assert bindings._objc is None

    # test missing accelerate attribute
    mock_accel = MagicMock()
    del mock_accel.cblas_sgemm
    with patch(
        "ctypes.CDLL", side_effect=[mock_accel, Exception(), Exception(), Exception(), Exception()]
    ):
        bindings._load_libraries()
        assert bindings._accelerate_lib is None

    bindings._load_libraries()
