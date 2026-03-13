"""Module docstring."""

import os
from pathlib import Path

# The base directory where JIT compiled extensions are stored.
# Can be overridden by the ONNX9000_CACHE_DIR environment variable.
ONNX9000_CACHE_DIR: Path = Path(
    os.environ.get("ONNX9000_CACHE_DIR", Path.home() / ".cache" / "onnx9000")
)

# Target C++ Compiler to use (e.g. g++, clang++).
# If empty, the build system will try to auto-detect.
ONNX9000_COMPILER: str = os.environ.get("ONNX9000_COMPILER", "")

# Emscripten compiler for web/wasm targets
ONNX9000_WASM_COMPILER: str = os.environ.get("ONNX9000_WASM_COMPILER", "emcc")

# Debug mode flag. If "1", keeps generated C++ files around.
ONNX9000_DEBUG: bool = os.environ.get("ONNX9000_DEBUG", "0") == "1"

# Apple Accelerate Framework switch for fast math and BLAS on macOS
ONNX9000_USE_ACCELERATE: bool = os.environ.get("ONNX9000_USE_ACCELERATE", "1") == "1"

# CUDA target for NVIDIA GPUs
ONNX9000_USE_CUDA: bool = os.environ.get("ONNX9000_USE_CUDA", "0") == "1"
ONNX9000_NVCC_COMPILER: str = os.environ.get("ONNX9000_NVCC_COMPILER", "nvcc")
