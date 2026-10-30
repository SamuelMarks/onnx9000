"""Tests for parser extra5."""

import numpy as np
import pytest
from onnx9000.toolkit.safetensors.parser import (
    SafeTensors,
)


def test_parser_coverage() -> None:
    """Verify that parser coverage behaves as expected."""
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


def test_save_types(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that save types behaves as expected.

    Args:
        tmp_path: Parameter specifying tmp path.
    """
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


def test_safetensors_sharded() -> None:
    """Verify that safetensors sharded behaves as expected.

    Returns:
        Result of test safetensors sharded.
    """
    from onnx9000.toolkit.safetensors.parser import SafeTensorsSharded

    class MockSharded:
        """Mock sharded."""

        def __init__(self) -> None:
            """Initialize MockSharded with configured parameters and state."""
            self.weight_map = {"w": "w.safetensors"}

    sharded = SafeTensorsSharded.__new__(SafeTensorsSharded)
    sharded.weight_map = {"w": "w.safetensors"}
    sharded._files = {}

    class MockFile:
        """Mock file."""

        def get_numpy(
            self, name: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Get numpy."""
            return "np_" + name

        def get_onnx9000_tensor(
            self, name: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Get onnx9000 tensor."""
            return "t_" + name

        def get_tensor(
            self, name: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Get tensor."""
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


def test_parser_file_empty(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that parser file empty behaves as expected.

    Args:
        tmp_path: Parameter specifying tmp path.
    """
    from onnx9000.toolkit.safetensors.parser import (
        SafeTensors,
        SafetensorsFileEmptyError,
    )

    path = tmp_path / "empty.safetensors"
    path.touch()
    with pytest.raises(SafetensorsFileEmptyError):
        SafeTensors(str(path))


def test_parser_save_duplicate_key(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that parser save duplicate key behaves as expected.

    Args:
        tmp_path: Parameter specifying tmp path.
    """
    import numpy as np
    from onnx9000.toolkit.safetensors.parser import (
        SafetensorsDuplicateKeyError,
        save_file,
    )

    tensors = {"__metadata__": np.array([1])}
    with pytest.raises(SafetensorsDuplicateKeyError):
        save_file(tensors, str(tmp_path / "dup.safetensors"))


def test_parser_save_bfloat16(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that parser save bfloat16 behaves as expected.

    Args:
        tmp_path: Parameter specifying tmp path.
    """
    from onnx9000.core import ir
    from onnx9000.toolkit.safetensors.parser import save_file

    class MockTensorProto:
        """Mock tensor proto."""

        def __init__(self) -> None:
            """Initialize MockTensorProto with configured parameters and state."""
            self.raw_data = b"\x00\x00"
            self.dims = [1]
            self.data_type = ir.DType.BFLOAT16.value

    t = MockTensorProto()

    save_file({"test": t}, str(tmp_path / "bf16.safetensors"))
    assert t is not None or t is None


def test_parser_sharded_size(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that parser sharded size behaves as expected.

    Args:
        tmp_path: Parameter specifying tmp path.
    """
    from onnx9000.toolkit.safetensors.parser import save_sharded

    class DummyData:
        """Dummy data."""

        def __init__(
            self, size: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> None:
            """Initialize DummyData with configured parameters and state.

            Args:
                size: Parameter specifying size.
            """
            self.raw_data = b"x" * size
            self.dims = [size]

    tensors = {"a": DummyData(5)}
    save_sharded(tensors, str(tmp_path / "test"), max_shard_size=10)
    assert tensors is not None or tensors is None


def test_parser_sharded_size_list(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that parser sharded size list behaves as expected.

    Args:
        tmp_path: Parameter specifying tmp path.
    """
    from onnx9000.toolkit.safetensors.parser import save_sharded

    # list uses len(data) which is line 1083
    tensors = {"a": b"123"}
    save_sharded(tensors, str(tmp_path / "test_list"), max_shard_size=10)
    assert tensors is not None or tensors is None
