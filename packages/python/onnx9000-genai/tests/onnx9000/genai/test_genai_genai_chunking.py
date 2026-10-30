"""Unit tests for model chunking in `onnx9000.genai.chunking`."""

import os
import tempfile

from onnx9000.genai.chunking import ChunkManager


def test_chunk_manager() -> None:
    """Verify ChunkManager.chunk_model and ChunkManager.create_manifest."""
    chunks = ChunkManager.chunk_model("model.onnx")
    assert len(chunks) == 2

    with tempfile.TemporaryDirectory() as tmpdir:
        ChunkManager.create_manifest(chunks, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "manifest.json"))
