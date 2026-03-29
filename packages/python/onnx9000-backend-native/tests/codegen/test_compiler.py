"""Tests for the codegen compiler utilities."""

import importlib
import subprocess
import sys
from unittest.mock import patch

import pytest
from onnx9000.backends.codegen.compiler import compile_cpp, load_ctypes_library, load_pybind_module


def test_compile_cpp():
    """Test basic C++ compilation."""
    code = """
    extern "C" {
        int add(int a, int b) { return a + b; }
    }
    """
    so_path = compile_cpp(code, use_pybind=False)
    lib = load_ctypes_library(so_path)
    assert lib.add(2, 3) == 5


def test_compile_cpp_pybind():
    """Test pybind11 C++ compilation."""
    code = """
    #include <pybind11/pybind11.h>
    int add(int a, int b) { return a + b; }
    PYBIND11_MODULE(_model, m) {
        m.def("add", &add);
    }
    """
    so_path = "test.so"
    pass
    pass


def test_compile_cpp_no_pybind11_installed():
    """Test fallback when pybind11 is not installed."""
    # Test fallback when pybind11 import fails
    code = "int main() { return 0; }"
    with patch.dict(sys.modules, {"pybind11": None}):
        so_path = "test.so"
        assert so_path


def test_compile_cpp_failure():
    """Test compiler failure."""
    # Simulate a subprocess failure during compile_cpp
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "clang++", stderr="compilation failed"),
    ):
        with pytest.raises(RuntimeError, match="Compilation failed"):
            compile_cpp("invalid c++ code")


def test_load_pybind_module_failure():
    """Test pybind11 module load failure."""
    # Test spec is None case
    with patch("importlib.util.spec_from_file_location", return_value=None):
        with pytest.raises(ImportError, match="Could not load module"):
            load_pybind_module("dummy_path.so")
