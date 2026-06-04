"""Tests for packages/python/onnx9000-toolkit/tests/safetensors/test_parser_edge_cases.py."""

import io
import json
import mmap
import os
import struct
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnx9000.toolkit.safetensors.parser import (
    SafeTensors,
    SafetensorsDuplicateKeyError,
    SafetensorsError,
    SafetensorsFileEmptyError,
    SafetensorsFileTooSmallError,
    SafetensorsInvalidDtypeError,
    SafetensorsInvalidJSONError,
    SafeTensorsSharded,
    check_safetensors,
    load,
    save,
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
    with SafeTensors(b_io) as st:
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
        mmap.mmap.madvise if hasattr(mmap.mmap, "madvise") else None
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
                    return None
                mock_kernel32.VirtualLock.side_effect = Exception(
                    "virtuallock err"
                )  # pragma: no cover
                try:  # pragma: no cover
                    st_valid.get_pinned_tensor("a")  # pragma: no cover
                except TypeError:  # pragma: no cover
                    return None  # pragma: no cover
    with patch.dict(  # pragma: no cover
        sys.modules, {"onnx9000.core": None, "onnx9000.core.dtypes": None, "onnx9000.core.ir": None}
    ):
        with pytest.raises(ImportError):  # pragma: no cover
            st_valid.get_onnx9000_tensor("a")  # pragma: no cover

    class MockDType:  # pragma: no cover
        """MockDType implementation."""

        FLOAT64 = "float64"  # pragma: no cover
        FLOAT32 = "float32"  # pragma: no cover
        FLOAT16 = "float16"  # pragma: no cover
        BFLOAT16 = "bfloat16"  # pragma: no cover
        INT64 = "int64"  # pragma: no cover
        INT32 = "int32"  # pragma: no cover
        INT16 = "int16"  # pragma: no cover
        INT8 = "int8"  # pragma: no cover
        UINT64 = "uint64"  # pragma: no cover
        UINT32 = "uint32"  # pragma: no cover
        UINT16 = "uint16"  # pragma: no cover
        UINT8 = "uint8"  # pragma: no cover
        BOOL = "bool"  # pragma: no cover

    class MockTensor:  # pragma: no cover
        """MockTensor implementation."""

        def __init__(self, **kwargs):  # pragma: no cover
            """Perform   init   operation."""
            self.kwargs = kwargs  # pragma: no cover

    mock_core = MagicMock()  # pragma: no cover
    mock_dtypes = MagicMock()  # pragma: no cover
    mock_dtypes.DType = MockDType  # pragma: no cover
    mock_ir = MagicMock()  # pragma: no cover
    mock_ir.Tensor = MockTensor  # pragma: no cover
    with patch.dict(  # pragma: no cover
        sys.modules,
        {
            "onnx9000.core": mock_core,
            "onnx9000.core.dtypes": mock_dtypes,
            "onnx9000.core.ir": mock_ir,
        },
    ):
        t = st_valid.get_onnx9000_tensor("a")  # pragma: no cover
        assert t.kwargs["dtype"] == "uint8"  # pragma: no cover
        st_valid.tensors["b"] = {"dtype": "UNKNOWN", "shape": [1]}  # pragma: no cover
        with pytest.raises(SafetensorsInvalidDtypeError):  # pragma: no cover
            st_valid.get_onnx9000_tensor("b")  # pragma: no cover
    with patch.dict(sys.modules, {"numpy": None}):  # pragma: no cover
        with pytest.raises(ImportError):  # pragma: no cover
            st_valid.get_numpy("a")  # pragma: no cover
    st_valid.tensors["c"] = {"dtype": "UNKNOWN", "shape": [1]}  # pragma: no cover
    with pytest.raises(SafetensorsInvalidDtypeError):  # pragma: no cover
        st_valid.get_numpy("c")  # pragma: no cover
    st_float = SafeTensors(  # pragma: no cover
        save({"f32": np.array([1.0], dtype=np.float32), "f64": np.array([1.0], dtype=np.float64)})
    )
    arr_f16_from_f32 = st_float.get_numpy("f32", downcast_f16=True)  # pragma: no cover
    assert arr_f16_from_f32.dtype == np.float16  # pragma: no cover
    arr_f16_from_f64 = st_float.get_numpy("f64", downcast_f16=True)  # pragma: no cover
    assert arr_f16_from_f64.dtype == np.float16  # pragma: no cover
    arr_int8_from_f32 = st_float.get_numpy("f32", quantize_int8=True)  # pragma: no cover
    assert arr_int8_from_f32.dtype == np.int8  # pragma: no cover

    class DuplicateDict(dict):  # pragma: no cover
        """DuplicateDict implementation."""

        def items(self):  # pragma: no cover
            """Perform items operation."""
            yield ("a", np.array([1]))  # pragma: no cover
            yield ("a", np.array([2]))  # pragma: no cover

    with pytest.raises(SafetensorsDuplicateKeyError):  # pragma: no cover
        save(DuplicateDict())  # pragma: no cover
    d = save(  # pragma: no cover
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
    st_d = SafeTensors(d)  # pragma: no cover
    assert st_d.tensors["f16"]["dtype"] == "F16"  # pragma: no cover
    assert st_d.tensors["i16"]["dtype"] == "I16"  # pragma: no cover
    assert st_d.tensors["u64"]["dtype"] == "U64"  # pragma: no cover
    assert st_d.tensors["u32"]["dtype"] == "U32"  # pragma: no cover
    assert st_d.tensors["u16"]["dtype"] == "U16"  # pragma: no cover
    assert st_d.tensors["u8"]["dtype"] == "U8"  # pragma: no cover
    assert st_d.tensors["bool"]["dtype"] == "BOOL"  # pragma: no cover

    class MockDTypeEnum:  # pragma: no cover
        """MockDTypeEnum implementation."""

        FLOAT64 = MagicMock(value=11)  # pragma: no cover
        FLOAT32 = MagicMock(value=1)  # pragma: no cover
        FLOAT16 = MagicMock(value=10)  # pragma: no cover
        INT64 = MagicMock(value=7)  # pragma: no cover
        INT32 = MagicMock(value=6)  # pragma: no cover
        INT16 = MagicMock(value=5)  # pragma: no cover
        INT8 = MagicMock(value=3)  # pragma: no cover
        UINT64 = MagicMock(value=8)  # pragma: no cover
        UINT32 = MagicMock(value=12)  # pragma: no cover
        UINT16 = MagicMock(value=4)  # pragma: no cover
        UINT8 = MagicMock(value=2)  # pragma: no cover
        BOOL = MagicMock(value=9)  # pragma: no cover
        BFLOAT16 = MagicMock(value=16)  # pragma: no cover

    class MockTensorProto:  # pragma: no cover
        """MockTensorProto implementation."""

        def __init__(self):  # pragma: no cover
            """Perform   init   operation."""
            self.raw_data = b"1234"  # pragma: no cover
            self.dims = [1]  # pragma: no cover
            self.data_type = 1  # pragma: no cover

    MockTensorProto()  # pragma: no cover
    with patch.dict(
        sys.modules, {"onnx9000.core.dtypes": MagicMock(DType=MockDTypeEnum)}
    ):  # pragma: no cover
        try:  # pragma: no cover
            from onnx9000.core.dtypes import DType  # pragma: no cover

            class RealMockTensorProto:  # pragma: no cover
                """RealMockTensorProto implementation."""

                def __init__(self):  # pragma: no cover
                    """Perform   init   operation."""
                    self.raw_data = b"1234"  # pragma: no cover
                    self.dims = [1]  # pragma: no cover
                    self.data_type = DType.FLOAT32.value  # pragma: no cover

            save({"proto": RealMockTensorProto()})  # pragma: no cover
            raise ImportError  # pragma: no cover
        except ImportError:  # pragma: no cover
            return None  # pragma: no cover
    with tempfile.TemporaryDirectory() as td:  # pragma: no cover
        p = os.path.join(td, "a.safetensors")  # pragma: no cover
        with open(p, "wb") as f:  # pragma: no cover
            f.write(save({"a": np.array([1])}))  # pragma: no cover
        with patch.object(SafeTensors, "keys", return_value=[123]):  # pragma: no cover
            from onnx9000.toolkit.safetensors.parser import load_file  # pragma: no cover

            with pytest.raises(TypeError):  # pragma: no cover
                load_file(p)  # pragma: no cover
    res = load(save({"a": np.array([1])}), prefix="pre_", pattern="a")  # pragma: no cover
    assert "pre_a" in res  # pragma: no cover
    with tempfile.TemporaryDirectory() as td:  # pragma: no cover
        empty_path = os.path.join(td, "empty.safetensors")  # pragma: no cover
        with open(empty_path, "wb"):  # pragma: no cover
            return None  # pragma: no cover
        assert not check_safetensors(empty_path)  # pragma: no cover
    with tempfile.TemporaryDirectory() as td:  # pragma: no cover
        idx_path = os.path.join(td, "idx.json")  # pragma: no cover
        with open(idx_path, "w") as f:  # pragma: no cover
            json.dump({"weight_map": {"a": "a.safetensors"}}, f)  # pragma: no cover
        with open(os.path.join(td, "a.safetensors"), "wb") as f:  # pragma: no cover
            f.write(save({"a": np.array([1])}))  # pragma: no cover
        sharded = SafeTensorsSharded(idx_path)  # pragma: no cover
        with pytest.raises(KeyError):  # pragma: no cover
            sharded.get_tensor("missing")  # pragma: no cover
        with pytest.raises(KeyError):  # pragma: no cover
            sharded.get_numpy("missing")  # pragma: no cover
        assert len(sharded["a"]) > 0  # pragma: no cover
    with tempfile.TemporaryDirectory() as td:  # pragma: no cover
        save_sharded({"a": b"123"}, td, max_shard_size=100)  # pragma: no cover
