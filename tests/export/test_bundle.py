import pytest
from pathlib import Path
import zipfile
from onnx9000.export.bundle import create_model_bundle


def test_create_model_bundle(tmp_path):
    output_path = tmp_path / "model.o9k"
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"onnx_content")
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text('{"model": "test"}')
    external_data_dir = tmp_path / "weights"
    external_data_dir.mkdir()
    bin_file1 = external_data_dir / "weight1.bin"
    bin_file1.write_bytes(b"weight1_data")
    bin_file2 = external_data_dir / "weight2.bin"
    bin_file2.write_bytes(b"weight2_data")
    create_model_bundle(output_path, onnx_file, external_data_dir, manifest_file)
    assert output_path.exists()
    with zipfile.ZipFile(output_path, "r") as zf:
        names = zf.namelist()
        assert "model.onnx" in names
        assert "manifest.json" in names
        assert "weights/weight1.bin" in names
        assert "weights/weight2.bin" in names
