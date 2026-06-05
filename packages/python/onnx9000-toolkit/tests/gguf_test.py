"""Tests for gguf test."""

import os
import struct
import tempfile

from onnx9000.toolkit.gguf.parser import GGUFError, GGUFParser


def write_string(f, s: str):
    """Docstring for D103."""
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def test_gguf_parser():
    """Docstring for D103."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        # Magic
        f.write(b"GGUF")
        # Version
        f.write(struct.pack("<I", 2))
        # Tensor count
        f.write(struct.pack("<Q", 1))
        # KV count
        f.write(struct.pack("<Q", 1))

        # KV Pairs
        write_string(f, "general.alignment")
        f.write(struct.pack("<I", 4))  # UINT32
        f.write(struct.pack("<I", 32))

        # Tensors
        write_string(f, "test_tensor")
        f.write(struct.pack("<I", 2))  # n_dims
        f.write(struct.pack("<Q", 2))  # dim 1
        f.write(struct.pack("<Q", 2))  # dim 2
        f.write(struct.pack("<I", 0))  # ggml_type = F32
        f.write(struct.pack("<Q", 0))  # offset

        # Padding to 32 alignment
        current_offset = f.tell()
        padding = (32 - (current_offset % 32)) % 32
        f.write(b"\0" * padding)

        # Tensor data (4 floats)
        f.write(struct.pack("<f", 1.0))
        f.write(struct.pack("<f", 2.0))
        f.write(struct.pack("<f", 3.0))
        f.write(struct.pack("<f", 4.0))

        filename = f.name

    try:
        with GGUFParser(filename) as parser:
            assert parser.magic == b"GGUF"
            assert parser.version == 2
            assert parser.metadata["general.alignment"] == 32
            assert "test_tensor" in parser.keys()
            tensor = parser.get_onnx9000_tensor("test_tensor")
            assert tensor.name == "test_tensor"
            assert tensor.shape == (2, 2)

            data = struct.unpack("<ffff", tensor.data)
            assert data == (1.0, 2.0, 3.0, 4.0)
    finally:
        os.remove(filename)


if __name__ == "__main__":
    test_gguf_parser()
    print("GGUF test passed!")
