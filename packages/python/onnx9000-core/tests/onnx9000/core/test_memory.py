"""Unit tests for zero-copy memory mapping in `onnx9000.core.memory`."""

import tempfile

import pytest
from onnx9000.core.memory import MemoryMapError, mmap_tensor_data


def test_memory_mmap() -> None:
    """Verify MemoryMapError and mmap_tensor_data."""
    err = MemoryMapError("map fail")
    assert "map fail" in str(err)

    with tempfile.NamedTemporaryFile() as tf:
        tf.write(b"0123456789abcdef")
        tf.flush()
        mv = mmap_tensor_data(tf.name, 4, 8)
        assert bytes(mv) == b"456789ab"

        with pytest.raises(MemoryMapError):
            mmap_tensor_data(tf.name, 0, 100)
