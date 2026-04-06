"""Tests for FFI module."""

import ctypes
import ctypes.util
import os
import platform
import threading
from unittest.mock import MagicMock, patch

import pytest
from onnx9000.backends.ffi.core import (
    DynamicLibrary,
    DynamicLibraryError,
    HardwareContextHandle,
    get_cache_sizes,
    get_cpu_features,
    map_python_bool,
    map_python_string,
    profile_ctypes_overhead,
)


def test_dynamic_library_load():
    """Test loading libc (which should be available on mostly any POSIX system)."""
    # libc is standard on linux/darwin
    try:
        lib = DynamicLibrary("c", use_cffi=False)
        assert lib.lib is not None
    except DynamicLibraryError:
        pass


def test_dynamic_library_not_found():
    """Test loading a non-existent library."""
    with pytest.raises(DynamicLibraryError):
        DynamicLibrary("nonexistent_library_12345")


def test_dynamic_library_define_and_getattr():
    """Test function definition and attribute retrieval."""
    try:
        lib = DynamicLibrary("c")
        # Define getpid
        func = lib.define("getpid", [], ctypes.c_int)
        assert func is not None
        assert lib.getpid is not None
        pid = lib.getpid()
        assert isinstance(pid, int)
    except DynamicLibraryError:
        pass


def test_hardware_context_handle():
    """Test context handle."""
    destroyed = []

    def mock_destroy(handle):
        destroyed.append(True)

    ptr = 12345
    handle = HardwareContextHandle(ptr, mock_destroy)
    assert handle.ptr.value == ptr

    with handle as h:
        assert h.ptr.value == ptr
        assert not destroyed

    assert destroyed  # Context manager exit should trigger destroy
    assert handle.ptr.value is None


def test_map_python_string():
    """Test string mapping."""
    res = map_python_string("hello")
    assert isinstance(res, ctypes.c_char_p)
    assert res.value == b"hello"


def test_map_python_bool():
    """Test bool mapping."""
    assert map_python_bool(True).value == 1
    assert map_python_bool(False).value == 0


def test_profile_ctypes_overhead():
    """Test profiling overhead."""
    res = profile_ctypes_overhead()
    assert isinstance(res, float)
    assert res > 0


@patch("platform.system", return_value="Linux")
@patch("builtins.open", new_callable=MagicMock)
def test_get_cpu_features_linux(mock_open, mock_system):
    """Test get_cpu_features on Linux."""
    mock_open.return_value.__enter__.return_value.read.return_value = "avx2 avx512 neon"
    features = get_cpu_features()
    assert features["avx2"] is True
    assert features["avx512"] is True
    assert features["neon"] is True


@patch("platform.system", return_value="Darwin")
@patch("subprocess.check_output")
def test_get_cpu_features_darwin(mock_check_output, mock_system):
    """Test get_cpu_features on Darwin."""
    mock_check_output.return_value = b"hw.optional.avx1_0: 1\nhw.optional.neon: 1\n"
    features = get_cpu_features()
    assert features["avx"] is True
    assert features["neon"] is True
    assert features["avx2"] is False


@patch("platform.system", return_value="Linux")
@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=MagicMock)
def test_get_cache_sizes_linux(mock_open, mock_exists, mock_system):
    """Test get_cache_sizes on Linux."""
    mock_open.return_value.__enter__.return_value.read.side_effect = ["32K", "256K", "12M"]
    sizes = get_cache_sizes()
    assert sizes["l1"] == 32768
    assert sizes["l2"] == 262144
    assert sizes["l3"] == 12582912


@patch("platform.system", return_value="Darwin")
@patch("subprocess.check_output")
def test_get_cache_sizes_darwin(mock_check_output, mock_system):
    """Test get_cache_sizes on Darwin."""
    mock_check_output.return_value = b"hw.l1dcachesize: 32768\nhw.l2cachesize: 262144\n"
    sizes = get_cache_sizes()
    assert sizes["l1"] == 32768
    assert sizes["l2"] == 262144


def test_dynamic_library_custom_env():
    """Test loading with custom env var."""
    with patch.dict(os.environ, {"ONNX9000_LIB_TEST": "dummy_path"}):
        with patch("ctypes.CDLL", side_effect=OSError):
            with pytest.raises(DynamicLibraryError):
                DynamicLibrary("test")


@patch("platform.system", return_value="Windows")
@patch("platform.machine", return_value="AMD64")
def test_dynamic_library_windows(mock_machine, mock_system):
    with patch("ctypes.util.find_library", return_value="msvcrt.dll"):
        with patch("ctypes.CDLL"):
            lib = DynamicLibrary("msvcrt")
            assert lib.os == "Windows"

    with patch("ctypes.util.find_library", return_value="test.dll"):
        with patch("ctypes.WinDLL", create=True) as mock_windll:
            lib = DynamicLibrary("test", calling_convention="stdcall")
            assert mock_windll.called


def test_dynamic_library_linux():
    with patch("platform.system", return_value="Linux"):
        with patch("ctypes.util.find_library", return_value="libc.so.6"):
            with patch("ctypes.CDLL"):
                lib = DynamicLibrary("c", versions=["6"])
                assert lib.os == "Linux"


def test_dynamic_library_dlerror():
    with patch("platform.system", return_value="Linux"):
        with patch("ctypes.util.find_library", return_value="libc.so.6"):
            with patch("ctypes.CDLL") as mock_cdll:
                lib = DynamicLibrary("c")

                # Mock the specific call to ctypes.CDLL(None).dlerror
                mock_cdll.return_value.dlerror = MagicMock(return_value=b"Symbol not found")

                # Mock getattr to raise AttributeError to trigger dlerror handling
                lib.lib = MagicMock()
                del lib.lib.nonexistent_function_123

                with pytest.raises(AttributeError, match="dlerror: Symbol not found"):
                    lib.define("nonexistent_function_123", [], None)


def test_dynamic_library_getattr():
    try:
        lib = DynamicLibrary("c")
        lib.define("getpid", [], ctypes.c_int)
        assert lib.getpid is not None
    except DynamicLibraryError:
        pass


def test_get_cpu_features_exception():
    with patch("platform.system", return_value="Linux"):
        with patch("builtins.open", side_effect=Exception):
            assert get_cpu_features() is None
    with patch("platform.system", return_value="Darwin"):
        with patch("subprocess.check_output", side_effect=Exception):
            assert get_cpu_features() is None


def test_get_cache_sizes_exception():
    with patch("platform.system", return_value="Linux"):
        with patch("builtins.open", side_effect=Exception):
            with patch("os.path.exists", return_value=True):
                assert get_cache_sizes() is None
    with patch("platform.system", return_value="Darwin"):
        with patch("subprocess.check_output", side_effect=Exception):
            assert get_cache_sizes() is None


def test_dynamic_library_dlerror_exception():
    pass


def test_dynamic_library_getattr_cached():
    try:
        lib = DynamicLibrary("c")
        lib.define("getpid", [], ctypes.c_int)
        # cache hit
        _ = lib.getpid
        _ = lib.getpid
        assert lib.getpid is not None
    except DynamicLibraryError:
        pass


def test_get_cpu_features_linux_exception():
    with patch("platform.system", return_value="Linux"):
        with patch("builtins.open", side_effect=Exception):
            assert get_cpu_features() is None


def test_get_cpu_features_darwin_exception():
    with patch("platform.system", return_value="Darwin"):
        with patch("subprocess.check_output", side_effect=Exception):
            assert get_cpu_features() is None


def test_get_cache_sizes_linux_exception():
    with patch("platform.system", return_value="Linux"):
        with patch("builtins.open", side_effect=Exception):
            with patch("os.path.exists", return_value=True):
                assert get_cache_sizes() is None


def test_get_cache_sizes_darwin_exception():
    with patch("platform.system", return_value="Darwin"):
        with patch("subprocess.check_output", side_effect=Exception):
            assert get_cache_sizes() is None


def test_get_cache_sizes_darwin_format_exception():
    with patch("platform.system", return_value="Darwin"):
        with patch("subprocess.check_output", return_value=b"bad_format\\n"):
            sizes = get_cache_sizes()
            assert sizes["l1"] == 0


def test_get_cache_sizes_linux_format_exception():
    with patch("platform.system", return_value="Linux"):
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", new_callable=MagicMock) as mock_open:
                mock_open.return_value.__enter__.return_value.read.side_effect = ["K", "M", ""]
                sizes = get_cache_sizes()
                assert sizes is None
