"""Tests for safetensors."""

import pytest
from onnx9000.tvm.relay.frontend.safetensors import load_safetensors_weights
import tempfile
import struct
import json


def test_load_safetensors_weights():
    """Test docstring."""
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        header_str = json.dumps({"test": "data"}).encode("utf-8")
        f.write(struct.pack("<Q", len(header_str)))
        f.write(header_str)
        fname = f.name
    res = load_safetensors_weights(fname)
    assert res == {"test": "data"}
