"""Module docstring."""

from pathlib import Path
from onnx9000.export.bundle import create_model_bundle


def test_create_model_bundle(tmp_path: Path):
    """test_create_model_bundle docstring."""
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_text("stub")

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text("{}")

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "chunk1.bin").write_text("stub")

    out = tmp_path / "bundle.o9k"
    create_model_bundle(out, onnx_file, weights_dir, manifest_file)

    assert out.exists()
