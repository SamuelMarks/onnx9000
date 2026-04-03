"""Final coverage gaps for Keras 2 H5 model parser."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We need to test lines 10-11 by forcing h5py to be None
# This is tricky because it's at module level.
# We can mock the import in a subprocess or use a trick.
# For now, let's target 120-121 and 129.


def test_keras_h5_parser_close_and_dataset():
    """Verify close() method and collect_weights with real h5py objects if possible."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")

    from onnx9000.converters.tf.keras_h5_parser import KerasH5Parser

    fname = "test_final.h5"
    with h5py.File(fname, "w") as f:
        f.attrs["model_config"] = json.dumps({"class_name": "Sequential", "config": {"layers": []}})
        weights_group = f.create_group("model_weights")
        l1 = weights_group.create_group("layer_1")
        l1.create_dataset("w0", data=np.array([1.0, 2.0]))

    try:
        parser = KerasH5Parser(filename=fname)
        # Test line 129
        parser.close()

        # Test 120-121 (collect_weights with isinstance(obj, h5py.Dataset))
        # We need a new parser because the old one is closed
        parser = KerasH5Parser(filename=fname)
        weights = parser.get_weights()
        assert "layer_1" in weights
        assert len(weights["layer_1"]) == 1
        assert np.allclose(weights["layer_1"][0], [1.0, 2.0])
        parser.close()
    finally:
        if os.path.exists(fname):
            os.remove(fname)


def test_keras_h5_parser_module_fallback():
    """Verify ImportError handling logic (simulated)."""
    # To cover line 22 and the 'h5py is None' path in __init__
    # we can manually set h5py to None in the module for a moment
    import onnx9000.converters.tf.keras_h5_parser as parser_mod

    old_h5 = parser_mod.h5py
    parser_mod.h5py = None
    try:
        from onnx9000.converters.tf.keras_h5_parser import KerasH5Parser

        with pytest.raises(ImportError, match="h5py is required"):
            KerasH5Parser()
    finally:
        parser_mod.h5py = old_h5


def test_keras_h5_parser_init_data():
    """Verify initialization with raw bytes data."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")

    from onnx9000.converters.tf.keras_h5_parser import KerasH5Parser

    # Create valid H5 in memory/disk and read as bytes
    fname = "test_bytes.h5"
    with h5py.File(fname, "w") as f:
        f.attrs["model_config"] = json.dumps({"class_name": "Sequential", "config": {"layers": []}})

    try:
        with open(fname, "rb") as f:
            data = f.read()

        # This will use io.BytesIO internally if KerasH5Parser supports it,
        # or we just check if it handles 'data' arg.
        parser = KerasH5Parser(data=data)
        graph = parser.parse()
        assert len(graph.nodes) == 0
        parser.close()
    finally:
        if os.path.exists(fname):
            os.remove(fname)
