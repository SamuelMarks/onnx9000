import os
from unittest.mock import patch

import pytest
from onnx9000.backends.rocm.compiler import ROCmCompiler


def test_rocm_compiler_cached(tmpdir) -> None:
    kernel_code = "void kernel() {}"
    cache_dir = str(tmpdir)
    hsaco_path = os.path.join(cache_dir, "test_kernel.hsaco")
    with open(hsaco_path, "wb") as f:
        f.write(b"cached_hsaco")

    res = ROCmCompiler.compile_kernel(kernel_code, "test_kernel", cache_dir=cache_dir)
    assert res == b"cached_hsaco"


def test_rocm_compiler_compile(tmpdir) -> None:
    kernel_code = "void kernel() {}"
    cache_dir = str(tmpdir)
    hsaco_path = os.path.join(cache_dir, "test_kernel2.hsaco")

    def fake_run(cmd, check, capture_output) -> None:
        with open(hsaco_path, "wb") as f:
            f.write(b"new_hsaco")

    with patch("subprocess.run", side_effect=fake_run):
        res = ROCmCompiler.compile_kernel(kernel_code, "test_kernel2", cache_dir=cache_dir)
        assert res == b"new_hsaco"


def test_rocm_compiler_hipcc_not_found(tmpdir) -> None:
    kernel_code = "void kernel() {}"
    cache_dir = str(tmpdir)
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        res = ROCmCompiler.compile_kernel(kernel_code, "test_kernel3", cache_dir=cache_dir)
        assert res == b""


def test_rocm_compiler_hipcc_failed(tmpdir) -> None:
    kernel_code = "void kernel() {}"
    cache_dir = str(tmpdir)
    import subprocess

    err = subprocess.CalledProcessError(1, "hipcc")
    err.stderr = b"error message"
    with patch("subprocess.run", side_effect=err):
        with pytest.raises(RuntimeError, match="ROCm Kernel compilation failed"):
            ROCmCompiler.compile_kernel(kernel_code, "test_kernel4", cache_dir=cache_dir)


def test_rocm_compiler_no_cache_dir() -> None:
    from unittest.mock import MagicMock

    with patch("tempfile.gettempdir", return_value="/tmp"):
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                res = ROCmCompiler.compile_kernel("void kernel() {}", "test_kernel")
                assert res is not None
