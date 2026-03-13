"""Module docstring."""

import os
from pathlib import Path
from onnx9000.export.chunking import (
    export_with_external_data,
    generate_chunk_manifest,
    compress_weights_to_int8,
    embed_metadata,
    calculate_sha256,
    merge_chunks,
    encrypt_chunk,
    decrypt_chunk,
)
from onnx9000.ir import Graph


def test_chunking_apis(tmp_path: Path):
    """test_chunking_apis docstring."""
    g = Graph("test")
    export_with_external_data(g, tmp_path)

    man = generate_chunk_manifest(g, {"tensor1": "chunk1.bin"})
    assert "chunk1.bin" in man

    compress_weights_to_int8(g)
    embed_metadata(g, "author", "1.0", "text")

    test_file = tmp_path / "test.bin"
    with open(test_file, "wb") as f:
        f.write(b"hello world")

    sha = calculate_sha256(test_file)
    assert sha == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    merge_chunks(tmp_path / "manifest.json", tmp_path / "merged.bin")
    encrypt_chunk(test_file, b"key")
    decrypt_chunk(test_file, b"key")
