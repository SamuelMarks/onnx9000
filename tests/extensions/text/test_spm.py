"""Module providing core logic and structural definitions."""

import struct
import pytest
from onnx9000.extensions.text.spm import parse_spm_model, _read_varint, SPMNode


def test_read_varint():
    """Provides semantic functionality and verification."""
    buf = bytes([10])
    val, off = _read_varint(buf, 0)
    assert val == 10
    assert off == 1
    buf2 = bytes([172, 2])
    val2, off2 = _read_varint(buf2, 0)
    assert val2 == 300
    assert off2 == 2


def test_parse_spm_model():
    """Provides semantic functionality and verification."""
    score_bytes = struct.pack("<f", -1.0)
    inner_msg = bytes([10, 2]) + b"hi" + bytes([21]) + score_bytes + bytes([24, 1])
    outer_msg = bytes([10, len(inner_msg)]) + inner_msg
    outer_msg += bytes([16, 5])
    outer_msg += bytes([25]) + b"\x00" * 8
    outer_msg += bytes([37]) + b"\x00" * 4
    outer_msg += bytes([42, 2]) + b"ok"
    pieces = parse_spm_model(outer_msg)
    assert len(pieces) == 1
    assert pieces[0].piece == "hi"
    assert pieces[0].score == -1.0
    assert pieces[0].type == 1


def test_parse_spm_model_inner_skips():
    """Provides semantic functionality and verification."""
    inner_msg = (
        bytes([10, 2])
        + b"hi"
        + bytes([32, 5])
        + bytes([45])
        + b"\x00" * 4
        + bytes([49])
        + b"\x00" * 8
        + bytes([58, 2])
        + b"no"
    )
    outer_msg = bytes([10, len(inner_msg)]) + inner_msg
    pieces = parse_spm_model(outer_msg)
    assert len(pieces) == 1
    assert pieces[0].piece == "hi"
