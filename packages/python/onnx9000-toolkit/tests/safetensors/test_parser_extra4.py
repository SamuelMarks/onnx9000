"""Tests for parser extra4."""

import os

import numpy as np
import pytest
from onnx9000.toolkit.safetensors.parser import SafeTensors, SafetensorsInvalidDtypeError


def test_parser_coverage():
    """Docstring for D103."""
    parser = SafeTensors.__new__(SafeTensors)
    parser.tensors = {
        "t1": {"dtype": "UNKN", "shape": [2, 2], "data_offsets": [0, 16]},
        "t2": {"dtype": "F64", "shape": [2, 2], "data_offsets": [0, 32]},
        "t3": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
    }
    parser.get_tensor = lambda name: b"\x00" * 32

    with pytest.raises(SafetensorsInvalidDtypeError):
        parser.get_numpy("t1")

    arr = parser.get_numpy("t2", downcast_f16=True)
    assert arr.dtype == np.float16

    parser.get_tensor = lambda name: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32).tobytes()
    arr2 = parser.get_numpy("t3", quantize_int8=True)
    assert arr2.dtype == np.int8
