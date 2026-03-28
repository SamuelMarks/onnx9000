"""Tests for packages/python/onnx9000-toolkit/tests/safetensors/test_parser_edge_cases.py."""

import os
import io
import mmap
import json
import struct
import sys
import ctypes
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from onnx9000.toolkit.safetensors.parser import (
    SafeTensors,
    SafetensorsError,
    SafetensorsFileEmptyError,
    SafetensorsFileTooSmallError,
    SafetensorsInvalidJSONError,
    SafetensorsDuplicateKeyError,
    SafetensorsInvalidDtypeError,
    save,
    load,
    check_safetensors,
    SafeTensorsSharded,
    save_sharded,
)


def test_parser_edge_cases_and_mocks():
    """Test parser edge cases and mocks."""
    with pytest.raises(SafetensorsFileEmptyError):
        SafeTensors(b"")
    with pytest.raises(SafetensorsFileTooSmallError):
        SafeTensors(b"123")
    b_io = io.BytesIO()
    b_io.write(save({"a": np.array([1, 2], dtype=np.int32)}))
    st = SafeTensors(b_io)
    assert "a" in st.keys()

    class StreamWithoutGetBuffer:
        """StreamWithoutGetBuffer implementation."""

        def __init__(self, data):
            """Perform   init   operation."""
            self.data = data
            self.pos = 0

        def read(self):
            """Perform read operation."""
            return self.data[self.pos :]

        def seek(self, offset, whence=0):
            """Perform seek operation."""
            if whence == 0:
                self.pos = offset
            elif whence == 2:
                self.pos = len(self.data) + offset

        def tell(self):
            """Perform tell operation."""
            return self.pos

    st_stream = SafeTensors(StreamWithoutGetBuffer(b_io.getvalue()))
    assert "a" in st_stream.keys()
    with pytest.raises(SafetensorsFileEmptyError):
        SafeTensors(StreamWithoutGetBuffer(b""))
    with pytest.raises(SafetensorsFileTooSmallError):
        SafeTensors(StreamWithoutGetBuffer(b"123"))
    with pytest.raises(TypeError):
        SafeTensors(12345)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "file.safetensors")
        with open(p, "wb") as f:
            f.write(b"x" * 10)
        with patch("os.open", side_effect=OSError("os error")):
            with pytest.raises(SafetensorsError, match="Failed to map file"):
                SafeTensors(p)
        with patch("mmap.mmap", side_effect=MemoryError("mem error")):
            with pytest.raises(SafetensorsError, match="insufficient address space"):
                SafeTensors(p)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "file.safetensors")
        with open(p, "wb") as f:
            f.write(save({"a": np.array([1, 2], dtype=np.int32)}))
        original_madvise = mmap.mmap.madvise if hasattr(mmap.mmap, "madvise") else None
        mock_mm = MagicMock()
        mock_mm.madvise.side_effect = Exception("madvise failed")
    with pytest.raises(SafetensorsInvalidJSONError, match="Tensor value must be a dict"):
        header = b'{"a": "not_a_dict"}'
        data = struct.pack("<Q", len(header)) + header + b"x"
        SafeTensors(data)
    with patch("json.loads", return_value={"__metadata__": {}}):
        with patch("json.loads") as mock_json:

            class DuplicateKeysDict(dict):
                """DuplicateKeysDict implementation."""

                def items(self):
                    """Perform items operation."""
                    yield ("a", {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]})
                    yield ("a", {"dtype": "I32", "shape": [1], "data_offsets": [4, 8]})

            mock_json.return_value = DuplicateKeysDict()
            header = b"{}"
            data = struct.pack("<Q", len(header)) + header + b"x" * 8
            with pytest.raises(SafetensorsDuplicateKeyError):
                SafeTensors(data)
    with pytest.raises(
        SafetensorsInvalidDtypeError, match="Complex types \\(C64\\) are not currently supported"
    ):
        header = b'{"a": {"dtype": "C64", "shape": [1], "data_offsets": [0, 8]}}'
        data = struct.pack("<Q", len(header)) + header + b"x" * 8
        SafeTensors(data)
    st_valid = SafeTensors(save({"a": np.array([1, 2], dtype=np.uint8)}))
    with pytest.raises(KeyError):
        st_valid.get_tensor("missing")
    st_empty = SafeTensors(save({"empty": np.array([], dtype=np.float32)}))
    assert len(st_empty.get_pinned_tensor("empty")) == 0
    with patch("sys.platform", "linux"):
        with patch("ctypes.CDLL") as mock_cdll:
            mock_libc = MagicMock()
            mock_libc.mlock.return_value = -1
            mock_cdll.return_value = mock_libc
            view = st_valid.get_pinned_tensor("a")
            assert len(view) > 0
        with patch("ctypes.CDLL", side_effect=Exception("cdll error")):
            view = st_valid.get_pinned_tensor("a")
    with patch("sys.platform", "win32"):
        with patch("mmap.mmap") as mock_mmap_win32:
            mock_mm_obj = MagicMock()
            mock_mmap_win32.return_value = mock_mm_obj
            with patch("ctypes.windll", create=True) as mock_windll:
                mock_kernel32 = MagicMock()
                mock_kernel32.VirtualLock.return_value = 0
                mock_windll.kernel32 = mock_kernel32
                try:
                    st_valid.get_pinned_tensor("a")
                except TypeError:
                    pass
                mock_kernel32.VirtualLock.side_effect = Exception("virtuallock err")
                try:
                    st_valid.get_pinned_tensor("a")
                except TypeError:
                    pass
    with patch.dict(
        sys.modules, {"onnx9000.core": None, "onnx9000.core.dtypes": None, "onnx9000.core.ir": None}
    ):
        with pytest.raises(ImportError):
            st_valid.get_onnx9000_tensor("a")

    class MockDType:
        """MockDType implementation."""

        FLOAT64 = "float64"
        FLOAT32 = "float32"
        FLOAT16 = "float16"
        BFLOAT16 = "bfloat16"
        INT64 = "int64"
        INT32 = "int32"
        INT16 = "int16"
        INT8 = "int8"
        UINT64 = "uint64"
        UINT32 = "uint32"
        UINT16 = "uint16"
        UINT8 = "uint8"
        BOOL = "bool"

    class MockTensor:
        """MockTensor implementation."""

        def __init__(self, **kwargs):
            """Perform   init   operation."""
            self.kwargs = kwargs

    mock_core = MagicMock()
    mock_dtypes = MagicMock()
    mock_dtypes.DType = MockDType
    mock_ir = MagicMock()
    mock_ir.Tensor = MockTensor
    with patch.dict(
        sys.modules,
        {
            "onnx9000.core": mock_core,
            "onnx9000.core.dtypes": mock_dtypes,
            "onnx9000.core.ir": mock_ir,
        },
    ):
        t = st_valid.get_onnx9000_tensor("a")
        assert t.kwargs["dtype"] == "uint8"
        st_valid.tensors["b"] = {"dtype": "UNKNOWN", "shape": [1]}
        with pytest.raises(SafetensorsInvalidDtypeError):
            st_valid.get_onnx9000_tensor("b")
    with patch.dict(sys.modules, {"numpy": None}):
        with pytest.raises(ImportError):
            st_valid.get_numpy("a")
    st_valid.tensors["c"] = {"dtype": "UNKNOWN", "shape": [1]}
    with pytest.raises(SafetensorsInvalidDtypeError):
        st_valid.get_numpy("c")
    st_float = SafeTensors(
        save({"f32": np.array([1.0], dtype=np.float32), "f64": np.array([1.0], dtype=np.float64)})
    )
    arr_f16_from_f32 = st_float.get_numpy("f32", downcast_f16=True)
    assert arr_f16_from_f32.dtype == np.float16
    arr_f16_from_f64 = st_float.get_numpy("f64", downcast_f16=True)
    assert arr_f16_from_f64.dtype == np.float16
    arr_int8_from_f32 = st_float.get_numpy("f32", quantize_int8=True)
    assert arr_int8_from_f32.dtype == np.int8

    class DuplicateDict(dict):
        """DuplicateDict implementation."""

        def items(self):
            """Perform items operation."""
            yield ("a", np.array([1]))
            yield ("a", np.array([2]))

    with pytest.raises(SafetensorsDuplicateKeyError):
        save(DuplicateDict())
    d = save(
        {
            "f16": np.array([1], dtype=np.float16),
            "i16": np.array([1], dtype=np.int16),
            "u64": np.array([1], dtype=np.uint64),
            "u32": np.array([1], dtype=np.uint32),
            "u16": np.array([1], dtype=np.uint16),
            "u8": np.array([1], dtype=np.uint8),
            "bool": np.array([True], dtype=np.bool_),
        }
    )
    st_d = SafeTensors(d)
    assert st_d.tensors["f16"]["dtype"] == "F16"
    assert st_d.tensors["i16"]["dtype"] == "I16"
    assert st_d.tensors["u64"]["dtype"] == "U64"
    assert st_d.tensors["u32"]["dtype"] == "U32"
    assert st_d.tensors["u16"]["dtype"] == "U16"
    assert st_d.tensors["u8"]["dtype"] == "U8"
    assert st_d.tensors["bool"]["dtype"] == "BOOL"

    class MockDTypeEnum:
        """MockDTypeEnum implementation."""

        FLOAT64 = MagicMock(value=11)
        FLOAT32 = MagicMock(value=1)
        FLOAT16 = MagicMock(value=10)
        INT64 = MagicMock(value=7)
        INT32 = MagicMock(value=6)
        INT16 = MagicMock(value=5)
        INT8 = MagicMock(value=3)
        UINT64 = MagicMock(value=8)
        UINT32 = MagicMock(value=12)
        UINT16 = MagicMock(value=4)
        UINT8 = MagicMock(value=2)
        BOOL = MagicMock(value=9)
        BFLOAT16 = MagicMock(value=16)

    class MockTensorProto:
        """MockTensorProto implementation."""

        def __init__(self):
            """Perform   init   operation."""
            self.raw_data = b"1234"
            self.dims = [1]
            self.data_type = 1

    MockTensorProto()
    with patch.dict(sys.modules, {"onnx9000.core.dtypes": MagicMock(DType=MockDTypeEnum)}):
        try:
            from onnx9000.core.dtypes import DType

            class RealMockTensorProto:
                """RealMockTensorProto implementation."""

                def __init__(self):
                    """Perform   init   operation."""
                    self.raw_data = b"1234"
                    self.dims = [1]
                    self.data_type = DType.FLOAT32.value

            save({"proto": RealMockTensorProto()})
            raise ImportError
        except ImportError:
            pass
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "a.safetensors")
        with open(p, "wb") as f:
            f.write(save({"a": np.array([1])}))
        with patch.object(SafeTensors, "keys", return_value=[123]):
            from onnx9000.toolkit.safetensors.parser import load_file

            with pytest.raises(TypeError):
                load_file(p)
    res = load(save({"a": np.array([1])}), prefix="pre_", pattern="a")
    assert "pre_a" in res
    with tempfile.TemporaryDirectory() as td:
        empty_path = os.path.join(td, "empty.safetensors")
        with open(empty_path, "wb"):
            pass
        assert check_safetensors(empty_path) == False
    with tempfile.TemporaryDirectory() as td:
        idx_path = os.path.join(td, "idx.json")
        with open(idx_path, "w") as f:
            json.dump({"weight_map": {"a": "a.safetensors"}}, f)
        with open(os.path.join(td, "a.safetensors"), "wb") as f:
            f.write(save({"a": np.array([1])}))
        sharded = SafeTensorsSharded(idx_path)
        with pytest.raises(KeyError):
            sharded.get_tensor("missing")
        with pytest.raises(KeyError):
            sharded.get_numpy("missing")
        assert len(sharded["a"]) > 0
    with tempfile.TemporaryDirectory() as td:
        save_sharded({"a": b"123"}, td, max_shard_size=100)
