"""Tests for parser extra3."""

from unittest.mock import patch

import pytest
from onnx9000.toolkit.safetensors.parser import SafeTensors


def test_import_errors() -> None:
    """Verify that import errors behaves as expected.

    Returns:
        Result of test import errors.

    Raises:
        ImportError: If an invalid state or argument is encountered during execution.
    """
    import builtins

    original_import = builtins.__import__

    def mock_import(
        name: int | float | str | bool | bytes | list | dict | tuple | None,
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Mock import."""
        if "onnx9000.core" in name or "numpy" in name:
            raise ImportError(f"Mocked ImportError for {name}")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        parser = SafeTensors.__new__(SafeTensors)
        with pytest.raises(ImportError):
            parser.get_onnx9000_tensor("test")

        with pytest.raises(ImportError):
            parser.get_numpy("test")
