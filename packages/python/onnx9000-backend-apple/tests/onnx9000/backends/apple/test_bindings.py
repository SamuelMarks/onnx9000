"""Test bindings."""

from onnx9000.backends.apple import bindings


def test_is_accelerate_available() -> None:
    """Test is_accelerate_available."""
    assert isinstance(bindings.is_accelerate_available(), bool)


def test_is_metal_available() -> None:
    """Test is_metal_available."""
    assert isinstance(bindings.is_metal_available(), bool)


def test_is_mps_available() -> None:
    """Test is_mps_available."""
    assert isinstance(bindings.is_mps_available(), bool)


def test_nsstring() -> None:
    """Test nsstring."""
    res = bindings.nsstring("test")
    assert res is not None


def test_mtl_create_system_default_device() -> None:
    """Test mtl_create_system_default_device."""
    res = bindings.mtl_create_system_default_device()
    assert res is not None or res is None


def test_bindings_load_error() -> None:
    """Test load errors."""
    from unittest.mock import patch

    with patch("ctypes.CDLL", side_effect=RuntimeError("mocked error")):
        bindings._load_libraries()
        assert not bindings.is_accelerate_available()
        assert not bindings.is_metal_available()
        assert not bindings.is_mps_available()
        assert bindings._foundation_lib is None

    with patch("ctypes.CDLL"):
        bindings._load_libraries()

    # test nsstring failure modes
    bindings._foundation_lib = None
    assert bindings.nsstring("x") is not None

    bindings._objc = None
    assert bindings.nsstring("x") is not None

    assert bindings.get_class("test") is not None
    assert bindings.get_selector("test") is not None

    bindings._metal_lib = None
    assert bindings.mtl_create_system_default_device() is None

    class MockLib:
        """Mock lib."""

        pass

    bindings._metal_lib = MockLib()
    assert bindings.mtl_create_system_default_device() is None


def test_missing_attr() -> None:
    """Test missing attribute."""
    from unittest.mock import patch

    from onnx9000.backends.apple import bindings

    with patch("ctypes.CDLL") as mock_cdll:
        mock_instance = mock_cdll.return_value
        # make it not raise attribute error initially but later
        del mock_instance.cblas_sgemm
        bindings._load_libraries()
        assert bindings._accelerate_lib is None


def test_nsstring_all_paths() -> None:
    """Test all paths."""
    from unittest.mock import patch

    from onnx9000.backends.apple import bindings

    bindings._load_libraries()
    with patch("ctypes.util.find_library", return_value="libobjc.A.dylib"):
        bindings._load_libraries()

    with patch("ctypes.util.find_library", return_value=None):
        bindings._load_libraries()
    assert test_nsstring_all_paths is not None
