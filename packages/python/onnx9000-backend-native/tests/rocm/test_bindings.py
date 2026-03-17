"""Tests the bindings module functionality."""

from unittest.mock import MagicMock

import onnx9000.backends.rocm.bindings as bindings
import pytest


def test_rocm_bindings_available() -> None:
    """Tests the rocm bindings available functionality."""
    # Because these are module level, they could be anything depending on test environment.
    # We just ensure they return a bool
    assert isinstance(bindings.is_hip_available(), bool)
    assert isinstance(bindings.is_rocblas_available(), bool)
    assert isinstance(bindings.is_miopen_available(), bool)


def test_register_apis() -> None:
    """Tests the register apis functionality."""
    mock_lib = MagicMock()
    bindings._register_hip_api(mock_lib)
    assert mock_lib.hipMalloc.argtypes is not None

    mock_roc = MagicMock()
    bindings._register_rocblas_api(mock_roc)
    assert mock_roc.rocblas_sgemm.argtypes is not None

    mock_mio = MagicMock()
    bindings._register_miopen_api(mock_mio)
    assert mock_mio.miopenConvolutionForward.argtypes is not None

    bindings._register_hip_api(None)
    bindings._register_rocblas_api(None)
    bindings._register_miopen_api(None)


def test_check_errors() -> None:
    """Tests the check errors functionality."""
    bindings.check_hip_error(0)
    bindings.check_rocblas_error(0)
    bindings.check_miopen_error(0)

    with pytest.raises(RuntimeError, match="HIP Error"):
        bindings.check_hip_error(1)

    with pytest.raises(RuntimeError, match="rocBLAS Error"):
        bindings.check_rocblas_error(1)

    with pytest.raises(RuntimeError, match="MIOpen Error"):
        bindings.check_miopen_error(1)
