"""Module providing core logic and structural definitions."""

import os
from pathlib import Path

ONNX9000_CACHE_DIR: Path = Path(
    os.environ.get("ONNX9000_CACHE_DIR", Path.home() / ".cache" / "onnx9000")
)
ONNX9000_COMPILER: str = os.environ.get("ONNX9000_COMPILER", "")
ONNX9000_WASM_COMPILER: str = os.environ.get("ONNX9000_WASM_COMPILER", "emcc")
ONNX9000_DEBUG: bool = os.environ.get("ONNX9000_DEBUG", "0") == "1"
ONNX9000_USE_ACCELERATE: bool = os.environ.get("ONNX9000_USE_ACCELERATE", "1") == "1"
ONNX9000_USE_CUDA: bool = os.environ.get("ONNX9000_USE_CUDA", "0") == "1"
ONNX9000_NVCC_COMPILER: str = os.environ.get("ONNX9000_NVCC_COMPILER", "nvcc")
