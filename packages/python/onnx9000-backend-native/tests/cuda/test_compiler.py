"""Tests the compiler module functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest
from onnx9000.backends.cuda.compiler import CUDACompiler


def test_cuda_compiler_cached(tmpdir) -> None:
    """Tests the cuda compiler cached functionality."""
    kernel_code = "void kernel() {}"
    cache_dir = str(tmpdir)
    ptx_path = os.path.join(cache_dir, "test_kernel.ptx")
    with open(ptx_path, "wb") as f:
        f.write(b"cached_ptx")

    res = CUDACompiler.compile_kernel(kernel_code, "test_kernel", cache_dir=cache_dir)
    assert res == b"cached_ptx"


def test_cuda_compiler_compile(tmpdir) -> None:
    """Tests the cuda compiler compile functionality."""
    kernel_code = "void kernel() {}"
    cache_dir = str(tmpdir)
    ptx_path = os.path.join(cache_dir, "test_kernel2.ptx")

    def fake_run(cmd, check, capture_output) -> None:
        """Tests the fake run functionality."""
        with open(ptx_path, "wb") as f:
            f.write(b"new_ptx")

    with patch("subprocess.run", side_effect=fake_run):
        res = CUDACompiler.compile_kernel(kernel_code, "test_kernel2", cache_dir=cache_dir)
        assert res == b"new_ptx"


def test_cuda_compiler_nvcc_not_found(tmpdir) -> None:
    """Tests the cuda compiler nvcc not found functionality."""
    kernel_code = "void kernel() {}"
    cache_dir = str(tmpdir)
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        res = CUDACompiler.compile_kernel(kernel_code, "test_kernel3", cache_dir=cache_dir)
        assert res == b""


def test_cuda_compiler_nvcc_failed(tmpdir) -> None:
    """Tests the cuda compiler nvcc failed functionality."""
    kernel_code = "void kernel() {}"
    cache_dir = str(tmpdir)
    import subprocess

    err = subprocess.CalledProcessError(1, "nvcc")
    err.stderr = b"error message"
    with patch("subprocess.run", side_effect=err):
        with pytest.raises(RuntimeError, match="CUDA Kernel compilation failed"):
            CUDACompiler.compile_kernel(kernel_code, "test_kernel4", cache_dir=cache_dir)


def test_calculate_grid_block() -> None:
    """Tests the calculate grid block functionality."""
    grid, block = CUDACompiler.calculate_grid_block(1000, 256)
    assert grid == 4
    assert block == 256


def test_cuda_compiler_no_cache_dir() -> None:
    """Tests the cuda compiler no cache dir functionality."""
    with patch("tempfile.gettempdir", return_value="/tmp"):
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                res = CUDACompiler.compile_kernel("void kernel() {}", "test_kernel")
                assert res is not None
