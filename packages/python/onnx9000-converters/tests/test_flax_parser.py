"""Tests for flax parser."""

import struct

import pytest
from onnx9000.converters.flax_parser import parse_msgpack


def test_parse_msgpack_basic():
    """Docstring for D103."""
    # True, False, None
    assert parse_msgpack(b"\xc3") is True
    assert parse_msgpack(b"\xc2") is False
    assert parse_msgpack(b"\xc0") is None


def test_parse_msgpack_ints():
    """Docstring for D103."""
    # positive fixint
    assert parse_msgpack(b"\x00") == 0
    assert parse_msgpack(b"\x7f") == 127
    # negative fixint
    assert parse_msgpack(b"\xff") == -1
    assert parse_msgpack(b"\xe0") == -32

    # uint8
    assert parse_msgpack(b"\xcc\x80") == 128
    # uint16
    assert parse_msgpack(b"\xcd\x01\x00") == 256
    # uint32
    assert parse_msgpack(b"\xce\x00\x01\x00\x00") == 65536
    # uint64
    assert parse_msgpack(b"\xcf\x00\x00\x00\x01\x00\x00\x00\x00") == 4294967296

    # int8
    assert parse_msgpack(b"\xd0\xdf") == -33
    # int16
    assert parse_msgpack(b"\xd1\xfe\x00") == -512
    # int32
    assert parse_msgpack(b"\xd2\xff\xfe\x00\x00") == -131072
    # int64
    assert parse_msgpack(b"\xd3\xff\xff\xff\xff\x00\x00\x00\x00") == -4294967296


def test_parse_msgpack_floats():
    """Docstring for D103."""
    assert parse_msgpack(b"\xca" + struct.pack(">f", 1.5)) == 1.5
    assert parse_msgpack(b"\xcb" + struct.pack(">d", 1.5)) == 1.5


def test_parse_msgpack_strings():
    """Docstring for D103."""
    # fixstr
    assert parse_msgpack(b"\xa1a") == "a"
    # str8
    s = "a" * 32
    assert parse_msgpack(b"\xd9\x20" + s.encode()) == s
    # str16
    s = "a" * 256
    assert parse_msgpack(b"\xda\x01\x00" + s.encode()) == s
    # str32
    s = "a" * 65536
    assert parse_msgpack(b"\xdb\x00\x01\x00\x00" + s.encode()) == s


def test_parse_msgpack_bin():
    """Docstring for D103."""
    # bin8
    b = b"a" * 32
    assert parse_msgpack(b"\xc4\x20" + b) == b
    # bin16
    b = b"a" * 256
    assert parse_msgpack(b"\xc5\x01\x00" + b) == b
    # bin32
    b = b"a" * 65536
    assert parse_msgpack(b"\xc6\x00\x01\x00\x00" + b) == b


def test_parse_msgpack_arrays():
    """Docstring for D103."""
    # fixarray
    assert parse_msgpack(b"\x92\xc3\xc2") == [True, False]
    # array16
    assert parse_msgpack(b"\xdc\x00\x02\xc3\xc2") == [True, False]
    # array32
    assert parse_msgpack(b"\xdd\x00\x00\x00\x02\xc3\xc2") == [True, False]


def test_parse_msgpack_maps():
    """Docstring for D103."""
    # fixmap
    assert parse_msgpack(b"\x81\xa1a\xc3") == {"a": True}
    # map16
    assert parse_msgpack(b"\xde\x00\x01\xa1a\xc3") == {"a": True}
    # map32
    assert parse_msgpack(b"\xdf\x00\x00\x00\x01\xa1a\xc3") == {"a": True}


def test_parse_msgpack_ext():
    """Docstring for D103."""
    # ext8
    assert parse_msgpack(b"\xc7\x01\x10\xff") == (0x10, b"\xff")
    # ext16
    assert parse_msgpack(b"\xc8\x00\x01\x10\xff") == (0x10, b"\xff")
    # ext32
    assert parse_msgpack(b"\xc9\x00\x00\x00\x01\x10\xff") == (0x10, b"\xff")
    # fixext1
    assert parse_msgpack(b"\xd4\x10\xff") == (0x10, b"\xff")
    # fixext2
    assert parse_msgpack(b"\xd5\x10\xff\xff") == (0x10, b"\xff\xff")
    # fixext4
    assert parse_msgpack(b"\xd6\x10" + b"\xff" * 4) == (0x10, b"\xff" * 4)
    # fixext8
    assert parse_msgpack(b"\xd7\x10" + b"\xff" * 8) == (0x10, b"\xff" * 8)
    # fixext16
    assert parse_msgpack(b"\xd8\x10" + b"\xff" * 16) == (0x10, b"\xff" * 16)


def test_parse_msgpack_unexpected_end():
    """Docstring for D103."""
    with pytest.raises(ValueError, match="Unexpected end of data"):
        parse_msgpack(b"\xc4")


def test_parse_msgpack_unimplemented():
    """Docstring for D103."""
    with pytest.raises(ValueError):
        parse_msgpack(b"\xc1")


def test_flax_parser_msgpack_unexpected_eof():
    """Docstring for D103."""
    import pytest
    from onnx9000.converters.flax_parser import parse_msgpack

    with pytest.raises(ValueError):
        parse_msgpack(b"")
