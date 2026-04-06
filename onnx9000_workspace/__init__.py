"""Workspace management for ONNX9000 models and configurations."""

import os


def setup_workspace(path: str) -> None:
    """Initialize a new ONNX9000 workspace."""
    os.makedirs(os.path.join(path, "models"), exist_ok=True)
    os.makedirs(os.path.join(path, "configs"), exist_ok=True)
    with open(os.path.join(path, "configs", "onnx9000.yaml"), "w") as f:
        f.write("version: 1.0\nmodels_dir: ../models\n")
    print(f"Workspace initialized at {path}")
