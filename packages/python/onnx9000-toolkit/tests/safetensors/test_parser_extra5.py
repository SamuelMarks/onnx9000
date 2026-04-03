import os

import numpy as np
import pytest
from onnx9000.toolkit.safetensors.parser import SafeTensors, SafetensorsInvalidDtypeError


def test_parser_coverage():
    parser = SafeTensors.__new__(SafeTensors)
    parser.tensors = {
        "t1": {"dtype": "F64", "shape": [2, 2], "data_offsets": [0, 32]},
    }

    parser.get_tensor = lambda name: np.zeros(4, dtype=np.float64).tobytes()
    arr = parser.get_numpy("t1", downcast_f16=True)
    assert arr.dtype == np.float16

    parser.tensors["t3"] = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    parser.get_tensor = lambda name: np.zeros(2, dtype=np.float32).tobytes()
    arr3 = parser.get_numpy("t3", downcast_f16=True)
    assert arr3.dtype == np.float16

    parser.tensors["t2"] = {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}
    parser.get_tensor = lambda name: np.array([1.0, 0.0], dtype=np.float16).tobytes()
    arr3 = parser.get_numpy("t2", quantize_int8=True)
    assert arr3.dtype == np.int8

    parser.get_tensor = lambda name: np.array([0.0, 0.0], dtype=np.float16).tobytes()
    arr4 = parser.get_numpy("t2", quantize_int8=True)
    assert arr4.dtype == np.int8


def test_save_types(tmp_path):
    from onnx9000.toolkit.safetensors.parser import save_file

    tensors = {
        "f64": np.zeros(2, dtype=np.float64),
        "f16": np.zeros(2, dtype=np.float16),
        "i64": np.zeros(2, dtype=np.int64),
        "i32": np.zeros(2, dtype=np.int32),
        "i16": np.zeros(2, dtype=np.int16),
        "u64": np.zeros(2, dtype=np.uint64),
        "u32": np.zeros(2, dtype=np.uint32),
        "u16": np.zeros(2, dtype=np.uint16),
        "u8": np.zeros(2, dtype=np.uint8),
        "b": np.zeros(2, dtype=bool),
    }
    # Empty shape
    tensors["empty"] = np.zeros((1,), dtype=np.float32)
    from onnx9000.toolkit.safetensors.parser import load_file

    # Save a file to test load_file
    save_file(tensors, str(tmp_path / "test3.safetensors"))

    # Mock keys to return non-string
    original_keys = SafeTensors.keys
    SafeTensors.keys = lambda self: [123]
    try:
        with pytest.raises(TypeError):
            load_file(str(tmp_path / "test3.safetensors"))
    finally:
        SafeTensors.keys = original_keys
    from onnx9000.toolkit.safetensors.parser import check_safetensors

    assert check_safetensors(str(tmp_path / "test3.safetensors")) is True

    with open(str(tmp_path / "bad.safetensors"), "wb") as f:
        f.write(b"\xff\xff")
    assert check_safetensors(str(tmp_path / "bad.safetensors")) is False


def test_safetensors_sharded():
    from onnx9000.toolkit.safetensors.parser import SafeTensorsSharded

    class MockSharded:
        def __init__(self):
            self.weight_map = {"w": "w.safetensors"}

    sharded = SafeTensorsSharded.__new__(SafeTensorsSharded)
    sharded.weight_map = {"w": "w.safetensors"}
    sharded._files = {}

    class MockFile:
        def get_numpy(self, name):
            return "np_" + name

        def get_onnx9000_tensor(self, name):
            return "t_" + name

        def get_tensor(self, name):
            return "b_" + name

    sharded._get_file = lambda name: MockFile()

    assert sharded.get_numpy("w") == "np_w"
    assert sharded.get_tensor("w") == "b_w"

    with pytest.raises(KeyError):
        sharded.get_numpy("bad")
    assert "w" in sharded
    assert "bad" not in sharded
    assert sharded["w"] == "b_w"

    with pytest.raises(KeyError):
        _ = sharded["bad"]
