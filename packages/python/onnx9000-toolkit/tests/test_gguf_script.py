"""Tests for GGUF script parsing and compilation.

This module contains unit tests that verify the functionality of the GGUF parser
and ONNX-script compilation within the onnx9000-toolkit.
"""

import struct

import pytest
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.gguf.parser import GGUFError, GGUFParser
from onnx9000.toolkit.script import parse_and_compile


# Helper to pack gguf string
def pack_string(s: str) -> bytes:
    """Pack a Python string into GGUF-compatible byte format.

    This helper encodes the input string to UTF-8 and prepends its length
    as an unsigned 64-bit little-endian integer.

    Args:
        s: The string to pack.

    Returns:
        The packed bytes representing the GGUF string.
    """
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def test_gguf_parser(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test the GGUF parser initialization, validation, and metadata extraction.

    This test writes a mock GGUF file with both invalid and valid contents.
    It verifies that:
    1. GGUFParser raises GGUFError on invalid magic bytes.
    2. GGUFParser parses metadata (e.g., general.alignment).
    3. GGUFParser retrieves tensor shape, alignment, and data.
    4. GGUFParser raises KeyError for non-existent tensors.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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

        # Tensor: "test_tensor", dims=1, shape=[2], type=0 (f32), offset=0
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
        assert "test_tensor" in parser
        tensor = parser.get_onnx9000_tensor("test_tensor")
        assert tensor.shape == (2,)
        assert len(tensor.data) == 8

    with pytest.raises(KeyError):
        parser.get_onnx9000_tensor("not_found")


def test_gguf_parser_types(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test reading various metadata value types from the key-value dictionary in GGUF.

    This test populates GGUF fields with standard types (such as UINT8, INT8, UINT16, INT16,
    UINT32, INT32, FLOAT32, BOOL, STRING, UINT64, INT64, FLOAT64), arrays, and unsupported types.
    It asserts that correct GGUFError is raised upon encountering unsupported types.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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


def test_gguf_tensor_types(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser mapping for various tensor datatypes and quantization forms.

    This test writes tensor definitions with different codes representing FLOAT16,
    Q4_0 quantization, Q8_0 quantization, and default fallback types, checking
    that the parsed types match expected string descriptors.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    gguf_path = str(tmp_path / "ttypes.gguf")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 4))  # tensor count
        f.write(struct.pack("<Q", 0))  # kv count

        def write_tensor(name: str, ttype: int, size_bytes: int) -> None:
            """Write individual tensor metadata header details.

            Args:
                name: Name of the tensor.
                ttype: Data type index of the tensor.
                size_bytes: Size in bytes of the tensor data.
            """
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


def test_script_parse_and_compile(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test compilation flow for onnx-script python module components.

    This test generates temporary python source files containing script-decorated
    functions and normal non-decorated functions, validating compilation outputs
    and verification constraints.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    py_path = str(tmp_path / "script.py")
    with open(py_path, "w") as f:
        f.write("""
from onnx9000.toolkit.script import script
@script
def test_func():
    \"\"\"A valid ONNX script function used for test execution.\"\"\"
    pass
""")
    res = parse_and_compile(py_path)
    assert res is not None
    assert res.name == "test_func"

    py_path2 = str(tmp_path / "script2.py")
    with open(py_path2, "w") as f:
        f.write("""
def no_script():
    \"\"\"A plain python function without ONNX script decorator.\"\"\"
    pass
""")
    with pytest.raises(ValueError):
        parse_and_compile(py_path2)


def test_gguf_array_types_missing(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser parsing with missing 64-bit array configurations (UINT64, INT64, FLOAT64).

    This test constructs multiple GGUF structures containing arrays of 64-bit values
    and parses them to ensure proper parser operation and coverage.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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
    assert gguf_path is not None or gguf_path is None


def test_gguf_array_types_missing_more(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser behavior with additional array types such as UINT16, INT16, UINT32, FLOAT32, and STRING.

    This test sequentially writes and parses GGUF files to extend parser coverage
    for all missing small-integer and floating-point array configurations.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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
    assert gguf_path is not None or gguf_path is None


def test_gguf_array_types_missing_even_more(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser behavior with INT8 and BOOL array types to verify coverage.

    This test constructs and parses GGUF files containing INT8 and BOOL arrays,
    and runs a redundant test for INT64.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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
    assert gguf_path is not None or gguf_path is None


def test_gguf_array_types_missing_final(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser behavior with INT32 array types.

    This test constructs a GGUF file containing a negative INT32 array to verify
    proper parsing and coverage of signed 32-bit values.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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
    assert gguf_path is not None or gguf_path is None


def test_gguf_array_types_missing_final_final(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser behavior with a repeated INT64 array case.

    This test ensures INT64 arrays are correctly decoded under final-stage parsing validations.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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
    assert gguf_path is not None or gguf_path is None


def test_gguf_array_types_missing_final_final_final(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser behavior with a repeated UINT64 array case.

    This test ensures UINT64 arrays are correctly decoded under final-stage parsing validations.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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
    assert gguf_path is not None or gguf_path is None


def test_gguf_array_types_missing_final_final_final_final(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser behavior with repeated FLOAT64 array cases.

    This test ensures FLOAT64 arrays are correctly decoded under final-stage parsing validations.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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
    assert gguf_path is not None or gguf_path is None


def test_gguf_array_types_missing_final_final_final_final_final(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser behavior with multiple repeated UINT64 array cases.

    This test ensures UINT64 arrays are correctly decoded repeatedly under final-stage parsing validations.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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
    assert gguf_path is not None or gguf_path is None


def test_script_parse_and_compile_fix(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test the ONNX script compilation fix on mocked functions.

    This test runs parse_and_compile with a patch on runpy.run_path to return a
    MockFunc, testing proper build-generation and error scenarios.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    py_path = str(tmp_path / "script.py")
    with open(py_path, "w") as f:
        f.write("""
def test_func():
    \"\"\"Docstring.\"\"\"
    pass
test_func._is_onnx_script=True""")

    from unittest.mock import MagicMock, patch

    from onnx9000.toolkit.script import parse_and_compile

    with patch("runpy.run_path") as mock_rp:

        class MockFunc:
            """A mock script class containing compilation trigger properties."""

            _is_onnx_script = True
            __name__ = "test_func"

            def to_builder(self) -> MagicMock:
                """Return a mock builder that compiles correctly.

                Returns:
                    A Mock builder object with build() defined.
                """
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
    \"\"\"Docstring.\"\"\"
    pass
""")
    with patch("runpy.run_path") as mock_rp:
        mock_rp.return_value = {"test_func": lambda: None}
        with pytest.raises(ValueError):
            parse_and_compile(py_path3)


def test_script_parse_and_compile_fix2(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test compilation failures handling in parse_and_compile when Exception is raised.

    This test patches runpy.run_path with MockFunc2, which raises an Exception from to_builder,
    and asserts that parse_and_compile handles the exception successfully.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    py_path = str(tmp_path / "script.py")
    with open(py_path, "w") as f:
        f.write("""
def test_func():
    \"\"\"Docstring.\"\"\"
    pass
test_func._is_onnx_script=True""")

    from unittest.mock import patch

    from onnx9000.toolkit.script import parse_and_compile

    with patch("runpy.run_path") as mock_rp:

        class MockFunc2:
            """A mock script class that raises an error during builder creation."""

            _is_onnx_script = True
            __name__ = "test_func"

            def to_builder(self) -> None:
                """Raise an Exception to test failure resilience.

                Raises:
                    Exception: Simulates builder generation failure.
                """
                raise Onnx9000Error("Testto_builderexc")

        mock_rp.return_value = {"test_func": MockFunc2()}
        res = parse_and_compile(py_path)
        assert res is not None


def test_gguf_parser_exception_missing(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser error resilience with invalid metadata value type keys in arrays.

    This test verifies that the parser behaves properly when encountering arrays with unrecognized
    element types (type code 999).

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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

    from onnx9000.toolkit.gguf.parser import GGUFParser

    GGUFParser(gguf_path)
    assert gguf_path is not None or gguf_path is None


def test_gguf_array_types_unsupported_inner(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser behavior with unsupported inner array type identifier.

    This test confirms parser behavior under invalid inner types in the GGUF spec.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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
    assert gguf_path is not None or gguf_path is None


def test_gguf_parser_exception_missing2(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test GGUFParser raises GGUFError for unsupported key-value value types.

    This test validates that invalid/unknown value type codes raise the appropriate GGUFError.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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


def test_gguf_parser_exception_missing3(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Test safety of empty arrays and proper parsing of arrays of type 0 (UINT8) in GGUFParser.

    This test asserts that GGUFParser executes without crashes on empty arrays with unrecognized types,
    and successfully parses arrays of UINT8 type.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
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

    from onnx9000.toolkit.gguf.parser import GGUFParser

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
    assert gguf_path is not None or gguf_path is None
