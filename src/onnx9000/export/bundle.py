"""
Model Bundling

Wraps the `.onnx` graph, chunked `.bin` weights, JSON manifest,
and specialized JS preprocessing scripts into a unified `.o9k` bundle
format (e.g. zip or tar) for easy distribution.
"""

import zipfile
from pathlib import Path


def create_model_bundle(
    output_path: Path, onnx_file: Path, external_data_dir: Path, manifest_file: Path
) -> None:
    """Creates a unified package containing all execution artifacts."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(onnx_file, arcname=onnx_file.name)
        zf.write(manifest_file, arcname=manifest_file.name)
        for bin_file in external_data_dir.glob("*.bin"):
            zf.write(bin_file, arcname=f"weights/{bin_file.name}")
