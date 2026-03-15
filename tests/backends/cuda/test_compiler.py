import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from onnx9000.backends.cuda.compiler import CUDACompiler
import onnx9000.backends.cuda.compiler as comp


def test_compiler_success():
    with patch("subprocess.run") as mock_run, tempfile.TemporaryDirectory() as tmp_dir:

        def fake_run(*args, **kwargs):
            out_file = args[0][args[0].index("-o") + 1]
            with open(out_file, "wb") as f:
                f.write(b"fake ptx content")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_run
        result = CUDACompiler.compile_kernel(
            "fake code", "test_kernel", cache_dir=tmp_dir
        )
        assert result == b"fake ptx content"
        mock_run.reset_mock()
        result_cached = CUDACompiler.compile_kernel(
            "fake code", "test_kernel", cache_dir=tmp_dir
        )
        assert result_cached == b"fake ptx content"
        mock_run.assert_not_called()


def test_compiler_nvcc_not_found():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = CUDACompiler.compile_kernel("fake code", "test_kernel2")
        assert result == b""


def test_compiler_nvcc_failure():
    import subprocess

    error = subprocess.CalledProcessError(1, ["nvcc"], stderr=b"syntax error")
    with patch("subprocess.run", side_effect=error):
        with pytest.raises(RuntimeError, match="CUDA Kernel compilation failed"):
            CUDACompiler.compile_kernel("fake code", "test_kernel3")


def test_calculate_grid_block():
    blocks, threads = CUDACompiler.calculate_grid_block(1000, 256)
    assert blocks == 4
    assert threads == 256
    blocks, threads = CUDACompiler.calculate_grid_block(256, 256)
    assert blocks == 1
    assert threads == 256
