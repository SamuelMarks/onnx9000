"""Unit tests and verification suite for jax all functionality."""

import struct

import numpy as np
import pytest
from onnx9000.jax.flax_parser import parse_msgpack
from onnx9000.jax.jax_importer import JAXImporter

# --- flax_parser.py ---


def pack_uint8(
    b: int | float | str | bool | bytes | list | dict | tuple | None,
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute pack uint8 operation.

    Args:
        b: Parameter specifying b.

    Returns:
        Result of pack uint8.
    """
    return struct.pack("B", b)


def pack_uint16(
    b: int | float | str | bool | bytes | list | dict | tuple | None,
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute pack uint16 operation.

    Args:
        b: Parameter specifying b.

    Returns:
        Result of pack uint16.
    """
    return struct.pack(">H", b)


def pack_uint32(
    b: int | float | str | bool | bytes | list | dict | tuple | None,
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute pack uint32 operation.

    Args:
        b: Parameter specifying b.

    Returns:
        Result of pack uint32.
    """
    return struct.pack(">I", b)


def pack_uint64(
    b: int | float | str | bool | bytes | list | dict | tuple | None,
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute pack uint64 operation.

    Args:
        b: Parameter specifying b.

    Returns:
        Result of pack uint64.
    """
    return struct.pack(">Q", b)


def pack_int8(
    b: int | float | str | bool | bytes | list | dict | tuple | None,
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute pack int8 operation.

    Args:
        b: Parameter specifying b.

    Returns:
        Result of pack int8.
    """
    return struct.pack(">b", b)


def pack_int16(
    b: int | float | str | bool | bytes | list | dict | tuple | None,
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute pack int16 operation.

    Args:
        b: Parameter specifying b.

    Returns:
        Result of pack int16.
    """
    return struct.pack(">h", b)


def pack_int32(
    b: int | float | str | bool | bytes | list | dict | tuple | None,
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute pack int32 operation.

    Args:
        b: Parameter specifying b.

    Returns:
        Result of pack int32.
    """
    return struct.pack(">i", b)


def pack_int64(
    b: int | float | str | bool | bytes | list | dict | tuple | None,
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute pack int64 operation.

    Args:
        b: Parameter specifying b.

    Returns:
        Result of pack int64.
    """
    return struct.pack(">q", b)


def pack_float(
    b: int | float | str | bool | bytes | list | dict | tuple | None,
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute pack float operation.

    Args:
        b: Parameter specifying b.

    Returns:
        Result of pack float.
    """
    return struct.pack(">f", b)


def pack_double(
    b: int | float | str | bool | bytes | list | dict | tuple | None,
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute pack double operation.

    Args:
        b: Parameter specifying b.

    Returns:
        Result of pack double.
    """
    return struct.pack(">d", b)


def test_parse_msgpack() -> None:
    # Test valid msgpack bytes
    """Verify that parse msgpack behaves as expected."""
    assert parse_msgpack(b"\x00") == 0
    assert parse_msgpack(b"\x7f") == 127
    assert parse_msgpack(b"\xe0") == -32
    assert parse_msgpack(b"\xff") == -1

    # Fixmap (empty)
    assert parse_msgpack(b"\x80") == {}
    assert parse_msgpack(b"\x81\xa1a\xc3") == {"a": True}
    # Fixarray (empty)
    assert parse_msgpack(b"\x90") == []
    assert parse_msgpack(b"\x91\xc2") == [False]
    # Fixstr (empty)
    assert parse_msgpack(b"\xa0") == ""
    # Fixstr "a"
    assert parse_msgpack(b"\xa1a") == "a"

    # Nil
    assert parse_msgpack(b"\xc0") is None
    # Bool
    assert parse_msgpack(b"\xc2") is False
    assert parse_msgpack(b"\xc3") is True

    # Bin 8, 16, 32
    assert parse_msgpack(b"\xc4\x01\xaa") == b"\xaa"
    assert parse_msgpack(b"\xc5\x00\x01\xaa") == b"\xaa"
    assert parse_msgpack(b"\xc6\x00\x00\x00\x01\xaa") == b"\xaa"

    # Float, Double
    assert np.isclose(parse_msgpack(b"\xca" + pack_float(1.23)), 1.23)
    assert np.isclose(parse_msgpack(b"\xcb" + pack_double(1.23)), 1.23)

    # Uint 8, 16, 32, 64
    assert parse_msgpack(b"\xcc" + pack_uint8(123)) == 123
    assert parse_msgpack(b"\xcd" + pack_uint16(1234)) == 1234
    assert parse_msgpack(b"\xce" + pack_uint32(123456)) == 123456
    assert parse_msgpack(b"\xcf" + pack_uint64(123456789)) == 123456789

    # Int 8, 16, 32, 64
    assert parse_msgpack(b"\xd0" + pack_int8(-12)) == -12
    assert parse_msgpack(b"\xd1" + pack_int16(-1234)) == -1234
    assert parse_msgpack(b"\xd2" + pack_int32(-123456)) == -123456
    assert parse_msgpack(b"\xd3" + pack_int64(-123456789)) == -123456789

    # Str 8, 16, 32
    assert parse_msgpack(b"\xd9\x01a") == "a"
    assert parse_msgpack(b"\xda\x00\x01a") == "a"
    assert parse_msgpack(b"\xdb\x00\x00\x00\x01a") == "a"

    # Array 16, 32
    assert parse_msgpack(b"\xdc\x00\x00") == []
    assert parse_msgpack(b"\xdd\x00\x00\x00\x00") == []

    # Map 16, 32
    assert parse_msgpack(b"\xde\x00\x00") == {}
    assert parse_msgpack(b"\xdf\x00\x00\x00\x00") == {}

    # Ext 8, 16, 32
    assert parse_msgpack(b"\xc7\x01\x05\xaa") == (5, b"\xaa")
    assert parse_msgpack(b"\xc8\x00\x01\x05\xaa") == (5, b"\xaa")
    assert parse_msgpack(b"\xc9\x00\x00\x00\x01\x05\xaa") == (5, b"\xaa")

    # Fixext 1, 2, 4, 8, 16
    assert parse_msgpack(b"\xd4\x05\xaa") == (5, b"\xaa")
    assert parse_msgpack(b"\xd5\x05\xaa\xbb") == (5, b"\xaa\xbb")
    assert parse_msgpack(b"\xd6\x05\xaa\xbb\xcc\xdd") == (5, b"\xaa\xbb\xcc\xdd")
    assert parse_msgpack(b"\xd7\x05" + b"\xaa" * 8) == (5, b"\xaa" * 8)
    assert parse_msgpack(b"\xd8\x05" + b"\xaa" * 16) == (5, b"\xaa" * 16)

    # Edge cases
    with pytest.raises(ValueError):
        parse_msgpack(b"")
    with pytest.raises(ValueError):
        parse_msgpack(b"\xc4")
    with pytest.raises(ValueError):
        parse_msgpack(b"\xd9")
    with pytest.raises(ValueError):
        parse_msgpack(b"\xc7")
    with pytest.raises(ValueError):
        parse_msgpack(b"\xd4")  # needs type and data

    with pytest.raises(ValueError, match="not implemented"):
        parse_msgpack(b"\xc1")


# --- jax_importer.py ---
def test_jax_importer() -> None:
    """Verify that jax importer behaves as expected.

    Returns:
        Result of test jax importer.
    """
    importer = JAXImporter()

    # Test var naming
    var1 = object()
    name1 = importer.get_var_name(var1)
    assert name1 == "v0"

    # Test var naming again
    name2 = importer.get_var_name(var1)
    assert name2 == "v0"

    # test unhashable list
    var2 = []
    name3 = importer.get_var_name(var2)
    assert name3 == "v1"

    # with aval
    class DummyAval:
        """Represents dummy aval."""

        def __init__(self) -> None:
            """Initialize DummyAval with configured parameters and state."""
            self.shape = (1, 2)
            self.dtype = np.float32

    class DummyVar:
        """Represents dummy var."""

        def __init__(self) -> None:
            """Initialize DummyVar with configured parameters and state."""
            self.aval = DummyAval()

    var3 = DummyVar()
    name4 = importer.get_var_name(var3)
    assert name4 == "v2"

    from onnx9000.core.dtypes import DType

    assert importer._map_dtype(np.float32) == DType.FLOAT32
    assert importer._map_dtype(np.int32) == DType.INT32
    assert importer._map_dtype(np.float64) == DType.FLOAT32  # fallback

    # We need to mock jax.make_jaxpr and everything around it to test import_func without JAX installed
    from unittest.mock import MagicMock

    class MockJaxpr:
        """Represents mock jaxpr."""

        def __init__(self) -> None:
            """Initialize MockJaxpr with configured parameters and state."""
            self.invars = [MagicMock()]
            self.outvars = [MagicMock()]
            self.constvars = [MagicMock()]

            eqn = MagicMock()
            eqn.invars = [self.invars[0]]
            eqn.outvars = [self.outvars[0]]
            eqn.params = {"test": 1}
            eqn.primitive.name = "mock_op"
            self.eqns = [eqn]

    class MockMadeJaxpr:
        """Represents mock made jaxpr."""

        def __init__(self) -> None:
            """Initialize MockMadeJaxpr with configured parameters and state."""
            self.jaxpr = MockJaxpr()
            self.consts = [np.array([1, 2])]

    def mock_make_jaxpr(
        f: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute mock make jaxpr operation.

        Args:
            f: Parameter specifying f.

        Returns:
            Result of mock make jaxpr.
        """

        def wrapper(
            *args: int | float | str | bool | bytes | list | dict | tuple | None,
            **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
        ) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Execute wrapper operation.

            Args:
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of wrapper.
            """
            f(*args, **kwargs)
            return MockMadeJaxpr()

        return wrapper

    import sys

    sys.modules["jax"] = MagicMock(make_jaxpr=mock_make_jaxpr)

    def dummy_func(
        x: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute dummy func operation.

        Args:
            x: Parameter specifying x.

        Returns:
            Result of dummy func.
        """
        return x

    graph = importer.import_func(dummy_func, 1)

    # graph should have the imported structure
    assert graph.name == "jax_graph"
    # constants added
    assert len(graph.tensors) >= 1
    # Check node was added
    assert len(graph.nodes) == 1

    node = graph.nodes[0]
    assert node.op_type == "Mock_op"  # fallback because "mock_op" not registered
    assert node.attributes == {"test": 1}

    # test mock_op registered
    from onnx9000.core.registry import global_registry

    def registered_op(
        inputs: int | float | str | bool | bytes | list | dict | tuple | None,
        outputs: int | float | str | bool | bytes | list | dict | tuple | None,
        params: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute registered op operation.

        Args:
            inputs: Parameter specifying inputs.
            outputs: Parameter specifying outputs.
            params: Parameter specifying params.

        Returns:
            Result of registered op.
        """
        from onnx9000.core.ir import Node

        return Node(
            op_type="RegisteredMockOp",
            inputs=inputs,
            outputs=outputs,
            attributes=params,
        )

    global_registry.register_op("jax", "mock_op")(registered_op)

    # clear and re-run
    importer2 = JAXImporter()
    graph2 = importer2.import_func(dummy_func, 1)
    assert graph2.nodes[0].op_type == "RegisteredMockOp"
