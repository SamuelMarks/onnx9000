"""Test coverage for Keras 2 H5 model parser - error cases and module fallback."""

import pytest
import sys
from unittest.mock import patch, MagicMock


def test_keras_h5_parser_import_error():
    """Verify ImportError when h5py is not available in __init__."""
    # This covers the check inside KerasH5Parser.__init__
    import onnx9000.converters.tf.keras_h5_parser as parser_mod

    old_h5 = parser_mod.h5py
    parser_mod.h5py = None
    try:
        from onnx9000.converters.tf.keras_h5_parser import KerasH5Parser

        with pytest.raises(ImportError, match="h5py is required"):
            KerasH5Parser()
    finally:
        parser_mod.h5py = old_h5


def test_keras_h5_parser_module_level_import_fail():
    """Verify module-level fallback when h5py cannot be imported (lines 10-11)."""
    # To truly cover the 'except ImportError' at the module level,
    # we need to re-import the module while h5py is blocked in sys.modules.
    if "onnx9000.converters.tf.keras_h5_parser" in sys.modules:
        del sys.modules["onnx9000.converters.tf.keras_h5_parser"]

    with patch.dict(sys.modules, {"h5py": None}):
        # Block h5py and re-import
        import onnx9000.converters.tf.keras_h5_parser as parser_mod

        assert parser_mod.h5py is None

    # Clean up for other tests
    if "onnx9000.converters.tf.keras_h5_parser" in sys.modules:
        del sys.modules["onnx9000.converters.tf.keras_h5_parser"]
