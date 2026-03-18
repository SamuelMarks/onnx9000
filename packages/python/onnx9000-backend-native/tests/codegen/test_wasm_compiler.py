"""Tests for the codegen compiler utilities."""

import os
import shutil
import subprocess
from unittest.mock import patch

import pytest
from onnx9000.backends.codegen.compiler import compile_wasm
from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor


def test_compile_add_graph_wasm():
    """Test compilation of a graph to WASM."""
    g = Graph("AddGraph")
    g.inputs = ["A", "B"]
    g.outputs = ["C"]

    g.tensors["A"] = Tensor("A", (2, 2), DType.FLOAT32)
    g.tensors["B"] = Tensor("B", (2, 2), DType.FLOAT32)
    g.tensors["C"] = Tensor("C", (2, 2), DType.FLOAT32)

    n = Node("Add", ["A", "B"], ["C"])
    g.nodes.append(n)

    gen = Generator(g)
    code = gen.generate()

    with (
        patch("subprocess.run"),
        patch("os.path.exists", return_value=True),
        patch("os.path.getsize", return_value=1024),
    ):
        # We mock subprocess.run so it doesn't actually try to run emcc

        compile_wasm(code)

        wasm_path_os = compile_wasm(code, opt_level="-Os")
        assert os.path.exists(wasm_path_os)

        wasm_path_oz = compile_wasm(code, opt_level="-Oz", standalone_wasm=True)
        assert os.path.exists(wasm_path_oz)
        assert wasm_path_oz.endswith(".wasm")

        wasm_path_node = compile_wasm(
            code,
            environment="node",
            initial_memory=16777216,
            emit_tsd=True,
            enable_simd=True,
            use_pthreads=True,
            maximum_memory=33554432,
        )
        assert os.path.exists(wasm_path_node)
        assert os.path.exists(wasm_path_node.replace(".js", ".d.ts"))


def test_compile_wasm_failed():
    """Test WASM compilation failure handling."""
    with patch(
        "subprocess.run", side_effect=subprocess.CalledProcessError(1, "emcc", stderr="error")
    ):
        with pytest.raises(RuntimeError, match="WASM Compilation failed"):
            compile_wasm("int main() { return 0; }")


def test_compile_wasm_not_found():
    """Test WASM compilation failure when emcc is not found."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="emcc not found"):
            compile_wasm("int main() { return 0; }")
