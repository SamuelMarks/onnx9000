"""Module docstring."""

from unittest.mock import patch

import pytest
from onnx9000.toolkit.safetensors.parser import SafeTensors


def test_import_errors():
    """Docstring for D103."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "onnx9000.core" in name or "numpy" in name:
            raise ImportError(f"Mocked ImportError for {name}")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        parser = SafeTensors.__new__(SafeTensors)
        with pytest.raises(ImportError):
            parser.get_onnx9000_tensor("test")

        with pytest.raises(ImportError):
            parser.get_numpy("test")
