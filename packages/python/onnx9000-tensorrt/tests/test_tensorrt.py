from unittest.mock import MagicMock, patch

import pytest
from onnx9000.tensorrt.ffi import TensorRTFFI, _phase_1_20_bindings


def test_ffi_init():
    with patch("ctypes.util.find_library", return_value=None):
        ffi = TensorRTFFI()
        assert ffi.lib is None
        ffi._extract_version()
        ffi._setup_logging_callback()


def test_ffi_mock_lib():
    mock_lib = MagicMock()
    mock_lib.getInferLibVersion.return_value = 10001

    with patch("ctypes.CDLL", return_value=mock_lib):
        ffi = TensorRTFFI()
        assert ffi.version == (10, 0, 1)

        # test callback
        if hasattr(ffi, "_c_log_callback"):
            pass

        ffi.register_pointer(123, "test")
        assert ffi.pointers[123] == "test"
        ffi.unregister_pointer(123)
        assert 123 not in ffi.pointers

        with pytest.raises(RuntimeError):
            ffi.check_error(1, "test error")
        ffi.check_error(0, "success")


def test_phase_1_20_bindings():
    assert _phase_1_20_bindings() is True


def test_ffi_coverage_gaps():
    from unittest.mock import MagicMock, patch

    from onnx9000.tensorrt.ffi import TensorRTFFI

    with patch("ctypes.util.find_library", return_value="dummy"):
        with patch("sys.platform", "win32"):
            with patch("ctypes.CDLL", side_effect=OSError("test win")):
                ffi = TensorRTFFI()

    with patch("ctypes.util.find_library", return_value="dummy"):
        with patch("sys.platform", "linux"):
            with patch("ctypes.CDLL", side_effect=OSError("test linux")):
                ffi = TensorRTFFI()

    # test extract version without getInferLibVersion
    mock_lib = MagicMock()
    del mock_lib.getInferLibVersion
    with patch("ctypes.CDLL", return_value=mock_lib):
        ffi = TensorRTFFI()

    # test callback with different severities
    mock_lib = MagicMock()
    mock_lib.getInferLibVersion.return_value = 10001
    with patch("ctypes.CDLL", return_value=mock_lib):
        ffi = TensorRTFFI()
        if hasattr(ffi, "_c_log_callback"):
            ffi._c_log_callback(None, 0, b"err")
            ffi._c_log_callback(None, 2, b"warn")
            ffi._c_log_callback(None, 3, b"info")
            ffi._c_log_callback(None, 4, b"verbose")
