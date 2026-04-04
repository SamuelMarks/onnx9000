"""Module docstring."""

import os
import struct

import pytest
from onnx9000.core.ir import DType, QuantizedTensor, Tensor
from onnx9000.toolkit.gguf.parser import GGUFError, GGUFParser


def pack_string(s: str) -> bytes:
    """Docstring for D103."""
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def build_gguf():
    """Docstring for D103."""
    # Magic
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)  # version
    out += struct.pack("<Q", 5)  # tensor count
    out += struct.pack("<Q", 24)  # kv count

    # Add all basic types
    def add_kv(k, vtype, pack_fmt, *vals):
        nonlocal out
        out += pack_string(k)
        out += struct.pack("<I", vtype)
        if pack_fmt:
            out += struct.pack(pack_fmt, *vals)

    add_kv("uint8", 0, "<B", 1)
    add_kv("int8", 1, "<b", -1)
    add_kv("uint16", 2, "<H", 1)
    add_kv("int16", 3, "<h", -1)
    add_kv("uint32", 4, "<I", 1)
    add_kv("int32", 5, "<i", -1)
    add_kv("float32", 6, "<f", 1.0)
    add_kv("bool", 7, "<?", True)

    # string
    out += pack_string("string_val")
    out += struct.pack("<I", 8)
    out += pack_string("hello")

    # Arrays 9
    def add_arr(k, atype, pack_fmt, vals):
        nonlocal out
        out += pack_string(k)
        out += struct.pack("<I", 9)
        out += struct.pack("<I", atype)
        out += struct.pack("<Q", len(vals))
        for v in vals:
            if atype == 8:
                out += pack_string(v)
            else:
                out += struct.pack(pack_fmt, v)

    add_arr("arr_u8", 0, "<B", [1])
    add_arr("arr_i8", 1, "<b", [-1])
    add_arr("arr_u16", 2, "<H", [1])
    add_arr("arr_i16", 3, "<h", [-1])
    add_arr("arr_u32", 4, "<I", [1])
    add_arr("arr_i32", 5, "<i", [-1])
    add_arr("arr_f32", 6, "<f", [1.0])
    add_arr("arr_b", 7, "<?", [True])
    add_arr("arr_str", 8, "", ["hi"])
    add_arr("arr_u64", 10, "<Q", [1])
    add_arr("arr_i64", 11, "<q", [-1])
    add_arr("arr_f64", 12, "<d", [1.0])

    add_kv("uint64", 10, "<Q", 1)
    add_kv("int64", 11, "<q", -1)
    add_kv("float64", 12, "<d", 1.0)

    # Write 5 tensors
    def add_tensor(name, n_dims, dims, ggml_type, offset):
        nonlocal out
        out += pack_string(name)
        out += struct.pack("<I", n_dims)
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", ggml_type)
        out += struct.pack("<Q", offset)

    add_tensor("t0", 1, [32], 0, 0)  # float32
    add_tensor("t1", 1, [32], 1, 128)  # float16
    add_tensor("t2", 1, [32], 2, 128 + 64)  # q4_0
    add_tensor("t8", 1, [32], 8, 128 + 64 + 18)  # q8_0
    add_tensor("t_unknown", 1, [32], 999, 128 + 64 + 18 + 34)  # fallback

    # Now write the data block to cover the bytes
    # Padding bytes to align to 32 bytes (default in gguf)
    pad = 32 - (len(out) % 32)
    out += b"\x00" * pad

    # 32 f32 = 128
    out += struct.pack("<32f", *([1.0] * 32))
    # 32 f16 = 64 (we'll just write 64 zeros)
    out += b"\x00" * 64
    # q4_0 (18 bytes)
    out += b"\x00" * 18
    # q8_0 (34 bytes)
    out += b"\x00" * 34
    # fallback 32 bytes
    out += b"\x00" * 32

    return out


def test_gguf_all_types(tmp_path):
    """Docstring for D103."""
    fpath = tmp_path / "test.gguf"
    with open(fpath, "wb") as f:
        f.write(build_gguf())

    parser = GGUFParser(str(fpath))
    assert parser.metadata["uint8"] == 1
    assert parser.metadata["int8"] == -1
    assert parser.metadata["uint16"] == 1
    assert parser.metadata["int16"] == -1
    assert parser.metadata["uint32"] == 1
    assert parser.metadata["int32"] == -1
    assert parser.metadata["float32"] == 1.0
    assert parser.metadata["bool"]
    assert parser.metadata["string_val"] == "hello"

    assert parser.metadata["arr_u8"] == [1]
    assert parser.metadata["arr_i8"] == [-1]
    assert parser.metadata["arr_u16"] == [1]
    assert parser.metadata["arr_i16"] == [-1]
    assert parser.metadata["arr_u32"] == [1]
    assert parser.metadata["arr_i32"] == [-1]
    assert parser.metadata["arr_f32"] == [1.0]
    assert parser.metadata["arr_b"] == [True]
    assert parser.metadata["arr_str"] == ["hi"]
    assert parser.metadata["arr_u64"] == [1]
    assert parser.metadata["arr_i64"] == [-1]
    assert parser.metadata["arr_f64"] == [1.0]

    assert parser.metadata["uint64"] == 1
    assert parser.metadata["int64"] == -1
    assert parser.metadata["float64"] == 1.0

    t0 = parser.get_onnx9000_tensor("t0")
    t1 = parser.get_onnx9000_tensor("t1")
    t2 = parser.get_onnx9000_tensor("t2")
    t8 = parser.get_onnx9000_tensor("t8")
    t_unknown = parser.get_onnx9000_tensor("t_unknown")

    assert t0.dtype == DType.FLOAT32
    assert t1.dtype == DType.FLOAT16
    assert isinstance(t2, QuantizedTensor)
    assert isinstance(t8, QuantizedTensor)
    assert t_unknown.dtype == DType.FLOAT32

    # Test __del__
    del parser


def test_gguf_unsupported(tmp_path):
    """Docstring for D103."""
    fpath = tmp_path / "test_err.gguf"
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)  # version
    out += struct.pack("<Q", 0)  # tensor count
    out += struct.pack("<Q", 1)  # kv count
    out += pack_string("bad")
    out += struct.pack("<I", 999)  # BAD TYPE

    with open(fpath, "wb") as f:
        f.write(out)

    with pytest.raises(GGUFError, match="Unsupported GGUF value type"):
        GGUFParser(str(fpath))


def test_gguf_tensor_not_found(tmp_path):
    """Docstring for D103."""
    fpath = tmp_path / "test.gguf"
    with open(fpath, "wb") as f:
        f.write(build_gguf())
    parser = GGUFParser(str(fpath))
    with pytest.raises(KeyError, match="Tensor missing not found"):
        parser.get_onnx9000_tensor("missing")


def test_file_not_found():
    """Docstring for D103."""
    with pytest.raises(FileNotFoundError):
        GGUFParser("invalid_path_42_xyz.gguf")


def test_gguf_context_and_keys(tmp_path):
    """Docstring for D103."""
    fpath = tmp_path / "test.gguf"
    with open(fpath, "wb") as f:
        f.write(build_gguf())
    with GGUFParser(str(fpath)) as parser:
        assert "t0" in parser.keys()


def test_gguf_bad_magic(tmp_path):
    """Docstring for D103."""
    fpath = tmp_path / "bad.gguf"
    with open(fpath, "wb") as f:
        f.write(b"BAD ")
    with pytest.raises(GGUFError, match="Invalid magic"):
        GGUFParser(str(fpath))


def test_gguf_buffer_error(tmp_path):
    """Docstring for D103."""
    fpath = tmp_path / "test.gguf"
    with open(fpath, "wb") as f:
        f.write(build_gguf())
    parser = GGUFParser(str(fpath))

    class MockMM:
        def close(self):
            raise BufferError("mock error")

    parser.mm = MockMM()
    parser.__exit__(None, None, None)
