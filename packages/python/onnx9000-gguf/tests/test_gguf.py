"""Comprehensive unit tests for onnx9000.gguf and onnx9000_gguf."""

import struct
from pathlib import Path

import onnx9000_gguf
import pytest
from onnx9000.gguf import GGUF, GGUFError, GGUFParser, parse_gguf, run


def test_gguf_run_and_parser(tmp_path: Path) -> None:
    """Verify run(), GGUF.process(), and parse_gguf() on a valid temporary GGUF file."""
    assert run() == "[gguf] processed"
    assert onnx9000_gguf.run() == "[gguf] processed"
    assert GGUF().process("model.gguf") == "GGUF processed model.gguf"

    gguf_file = tmp_path / "mini.gguf"
    header = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)
    gguf_file.write_bytes(header)

    with parse_gguf(str(gguf_file)) as parser:
        assert isinstance(parser, GGUFParser)
        assert parser.version == 3
        assert parser.keys() == []

    bad_file = tmp_path / "bad.gguf"
    bad_file.write_bytes(b"NOPE" + struct.pack("<IQQ", 3, 0, 0))
    with pytest.raises(GGUFError, match="Invalid magic bytes"):
        parse_gguf(str(bad_file))
