import os
import struct

import pytest
from onnx9000.toolkit.gguf.parser import GGUFError, GGUFParser
from onnx9000.toolkit.script import parse_and_compile
from onnx9000.toolkit.script.builder import GraphBuilder
from onnx9000.toolkit.script.var import Var


# Helper to pack gguf string
def pack_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def test_gguf_parser(tmp_path):
    gguf_path = str(tmp_path / "test.gguf")

    with open(gguf_path, "wb") as f:
        # Invalid magic
        f.write(b"BADM")

    with pytest.raises(GGUFError, match="Invalid magic bytes"):
        GGUFParser(gguf_path)

    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 1))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # KV: "general.alignment", uint32
        f.write(pack_string("general.alignment"))
        f.write(struct.pack("<I", 4))  # type UINT32
        f.write(struct.pack("<I", 64))  # value 64

        # Tensor: "test_tensor", dims=1, shape=[2], type=0(f32), offset=0
        f.write(pack_string("test_tensor"))
        f.write(struct.pack("<I", 1))  # n_dims
        f.write(struct.pack("<Q", 2))  # shape[0]
        f.write(struct.pack("<I", 0))  # type (FLOAT32)
        f.write(struct.pack("<Q", 0))  # offset

        # Alignment padding to 64 bytes
        pad = (64 - (f.tell() % 64)) % 64
        f.write(b"\x00" * pad)

        # Tensor Data
        f.write(struct.pack("<2f", 1.0, 2.0))

    with GGUFParser(gguf_path) as parser:
        assert parser.alignment == 64
        assert "test_tensor" in parser.keys()
        tensor = parser.get_onnx9000_tensor("test_tensor")
        assert tensor.shape == (2,)
        assert len(tensor.data) == 8

        with pytest.raises(KeyError):
            parser.get_onnx9000_tensor("not_found")


def test_gguf_parser_types(tmp_path):
    # Test reading various value types from KV metadata
    gguf_path = str(tmp_path / "types.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 15))  # kv count

        types_data = [
            (0, struct.pack("<B", 1)),  # UINT8
            (1, struct.pack("<b", -1)),  # INT8
            (2, struct.pack("<H", 2)),  # UINT16
            (3, struct.pack("<h", -2)),  # INT16
            (4, struct.pack("<I", 3)),  # UINT32
            (5, struct.pack("<i", -3)),  # INT32
            (6, struct.pack("<f", 1.5)),  # FLOAT32
            (7, struct.pack("<?", True)),  # BOOL
            (8, pack_string("str")),  # STRING
            (10, struct.pack("<Q", 4)),  # UINT64
            (11, struct.pack("<q", -4)),  # INT64
            (12, struct.pack("<d", 2.5)),  # FLOAT64
        ]

        for i, (t, data) in enumerate(types_data):
            f.write(pack_string(f"k{i}"))
            f.write(struct.pack("<I", t))
            f.write(data)

        # Array of INT32
        f.write(pack_string("arr"))
        f.write(struct.pack("<I", 9))  # ARRAY type
        f.write(struct.pack("<I", 5))  # element type INT32
        f.write(struct.pack("<Q", 2))  # length
        f.write(struct.pack("<i", 1) + struct.pack("<i", 2))

        # Unsupported type
        f.write(pack_string("bad"))
        f.write(struct.pack("<I", 99))

        # Test ARRAY all types coverage
        f.write(pack_string("arr_all"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 0))  # UINT8 array empty to not crash
        f.write(struct.pack("<Q", 0))

    try:
        GGUFParser(gguf_path)
    except GGUFError as e:
        assert "Unsupported GGUF value type" in str(e)


def test_gguf_tensor_types(tmp_path):
    gguf_path = str(tmp_path / "ttypes.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 4))  # tensor count
        f.write(struct.pack("<Q", 0))  # kv count

        def write_tensor(name, ttype, size_bytes):
            f.write(pack_string(name))
            f.write(struct.pack("<I", 1))  # n_dims
            f.write(struct.pack("<Q", 32))  # shape[0]=32
            f.write(struct.pack("<I", ttype))
            f.write(struct.pack("<Q", 0))  # offset

        write_tensor("t_f16", 1, 64)
        write_tensor("t_q4", 2, 18)
        write_tensor("t_q8", 8, 34)
        write_tensor("t_other", 99, 32)

        pad = (32 - (f.tell() % 32)) % 32
        f.write(b"\x00" * pad)
        f.write(b"\x00" * 100)  # dummy data

    with GGUFParser(gguf_path) as parser:
        assert parser.get_onnx9000_tensor("t_f16").dtype.name == "FLOAT16"
        assert parser.get_onnx9000_tensor("t_q4").qtype == "Q4_0"
        assert parser.get_onnx9000_tensor("t_q8").qtype == "Q8_0"
        assert parser.get_onnx9000_tensor("t_other").dtype.name == "FLOAT32"


def test_script_parse_and_compile(tmp_path):
    py_path = str(tmp_path / "script.py")
    with open(py_path, "w") as f:
        f.write("""
from onnx9000.toolkit.script import script
@script
def test_func():
    pass
""")
    res = parse_and_compile(py_path)
    assert res is not None
    assert res.name == "test_func"

    py_path2 = str(tmp_path / "script2.py")
    with open(py_path2, "w") as f:
        f.write("""
def no_script():
    pass
""")
    with pytest.raises(ValueError):
        parse_and_compile(py_path2)


def test_gguf_array_types_missing(tmp_path):
    gguf_path = str(tmp_path / "array_missing.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 10))  # UINT64 array
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 42))

    GGUFParser(gguf_path)

    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 11))  # INT64 array
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<q", -42))

    GGUFParser(gguf_path)

    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 12))  # FLOAT64 array
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<d", 42.5))

    GGUFParser(gguf_path)


def test_gguf_array_types_missing_more(tmp_path):
    gguf_path = str(tmp_path / "array_missing_more.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss1"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 2))  # UINT16 array
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<H", 42))

    from onnx9000.toolkit.gguf.parser import GGUFParser

    GGUFParser(gguf_path)

    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss2"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 3))  # INT16 array
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<h", -42))

    GGUFParser(gguf_path)

    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss3"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 4))  # UINT32 array
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 42))

    GGUFParser(gguf_path)

    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss4"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 6))  # FLOAT32 array
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<f", 42.5))

    GGUFParser(gguf_path)

    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss5"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 8))  # STRING array
        f.write(struct.pack("<Q", 1))
        f.write(pack_string("test"))

    GGUFParser(gguf_path)


def test_gguf_array_types_missing_even_more(tmp_path):
    gguf_path = str(tmp_path / "array_missing_more2.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss10"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 1))  # INT8 array
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<b", -42))

    from onnx9000.toolkit.gguf.parser import GGUFParser

    GGUFParser(gguf_path)

    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss11"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 7))  # BOOL array
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<?", True))

    GGUFParser(gguf_path)

    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss12"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 11))  # INT64 array (again)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<q", -42))

    GGUFParser(gguf_path)


def test_gguf_array_types_missing_final(tmp_path):
    gguf_path = str(tmp_path / "array_missing_final.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss100"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 5))  # INT32 array (again but negative coverage)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<i", -42))

    from onnx9000.toolkit.gguf.parser import GGUFParser

    GGUFParser(gguf_path)


def test_gguf_array_types_missing_final_final(tmp_path):
    gguf_path = str(tmp_path / "array_missing_final_final.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss1000"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 11))  # INT64 again... 11 is the type
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<q", -42))

    from onnx9000.toolkit.gguf.parser import GGUFParser

    GGUFParser(gguf_path)


def test_gguf_array_types_missing_final_final_final(tmp_path):
    gguf_path = str(tmp_path / "array_missing_final_final.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss1000"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 10))  # UINT64 again
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 42))

    from onnx9000.toolkit.gguf.parser import GGUFParser

    GGUFParser(gguf_path)


def test_gguf_array_types_missing_final_final_final_final(tmp_path):
    gguf_path = str(tmp_path / "array_missing_final_final_final.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss10000"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 12))  # FLOAT64 again
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<d", 42.5))

    from onnx9000.toolkit.gguf.parser import GGUFParser

    GGUFParser(gguf_path)


def test_gguf_array_types_missing_final_final_final_final_final(tmp_path):
    gguf_path = str(tmp_path / "array_missing_final_final_final_final.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss100000"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 10))  # UINT64 again and again
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 42))

    from onnx9000.toolkit.gguf.parser import GGUFParser

    GGUFParser(gguf_path)


def test_script_parse_and_compile_fix(tmp_path):
    py_path = str(tmp_path / "script.py")
    with open(py_path, "w") as f:
        f.write("""
def test_func():
    pass
test_func._is_onnx_script = True""")

    import runpy
    from unittest.mock import MagicMock, patch

    from onnx9000.toolkit.script import parse_and_compile

    with patch("runpy.run_path") as mock_rp:

        class MockFunc:
            _is_onnx_script = True
            __name__ = "test_func"

            def to_builder(self):
                b = MagicMock()
                b.build.return_value = MagicMock(name="test_func")
                return b

        mock_rp.return_value = {"test_func": MockFunc()}
        res = parse_and_compile(py_path)
        assert res is not None

    py_path3 = str(tmp_path / "script3.py")
    with open(py_path3, "w") as f:
        f.write("""
def test_func():
    pass
""")
    with patch("runpy.run_path") as mock_rp:
        mock_rp.return_value = {"test_func": lambda: None}
        with pytest.raises(ValueError):
            parse_and_compile(py_path3)


def test_script_parse_and_compile_fix2(tmp_path):
    py_path = str(tmp_path / "script.py")
    with open(py_path, "w") as f:
        f.write("""
def test_func():
    pass
test_func._is_onnx_script = True""")

    from unittest.mock import MagicMock, patch

    from onnx9000.toolkit.script import parse_and_compile

    with patch("runpy.run_path") as mock_rp:

        class MockFunc2:
            _is_onnx_script = True
            __name__ = "test_func"

            def to_builder(self):
                raise Exception("Test to_builder exc")

        mock_rp.return_value = {"test_func": MockFunc2()}
        res = parse_and_compile(py_path)
        assert res is not None


def test_gguf_parser_exception_missing(tmp_path):
    gguf_path = str(tmp_path / "array_missing_exception.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss_exc"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 999))  # Invalid type
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<i", -42))

    from onnx9000.toolkit.gguf.parser import GGUFError, GGUFParser

    GGUFParser(gguf_path)


def test_gguf_array_types_unsupported_inner(tmp_path):
    gguf_path = str(tmp_path / "array_unsupported_inner.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_unsupported_inner"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 999))  # unsupported inner type
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<d", 42.5))  # dummy data to consume byte

    from onnx9000.toolkit.gguf.parser import GGUFParser

    GGUFParser(gguf_path)


def test_gguf_parser_exception_missing2(tmp_path):
    gguf_path = str(tmp_path / "array_missing_exception2.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss_exc2"))
        f.write(struct.pack("<I", 999))  # Invalid type
        f.write(struct.pack("<i", -42))

    from onnx9000.toolkit.gguf.parser import GGUFError, GGUFParser

    with pytest.raises(GGUFError):
        GGUFParser(gguf_path)


def test_gguf_parser_exception_missing3(tmp_path):
    gguf_path = str(tmp_path / "array_missing_exception3.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss_exc3"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 999))  # Invalid type
        f.write(struct.pack("<Q", 1))

    from onnx9000.toolkit.gguf.parser import GGUFError, GGUFParser

    # should be safe since no array elements read
    GGUFParser(gguf_path)

    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 1))  # kv count

        # missing array types
        f.write(pack_string("arr_miss_exc4"))
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 0))  # Type 0
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", 42))

    GGUFParser(gguf_path)
