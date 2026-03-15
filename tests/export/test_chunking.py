import pytest
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
from onnx9000.core.ir import Graph


def test_chunking_stubs(tmp_path):
    graph = Graph("test")
    export_with_external_data(graph, tmp_path)
    compress_weights_to_int8(graph)
    embed_metadata(graph, "test_author", "1.0", "vision")
    merge_chunks(tmp_path / "manifest.json", tmp_path / "out.bin")
    encrypt_chunk(tmp_path / "chunk.bin", b"key")
    decrypt_chunk(tmp_path / "chunk.bin", b"key")


def test_calculate_sha256(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"hello world")
    sha = calculate_sha256(f)
    assert len(sha) == 64


def test_generate_chunk_manifest():
    graph = Graph("test")
    manifest = generate_chunk_manifest(graph, {"w1": "chunk1.bin"})
    assert "version" in manifest
    assert "w1" in manifest
