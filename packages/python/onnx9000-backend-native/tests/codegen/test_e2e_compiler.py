"""Tests for the codegen compiler utilities."""

import os

import numpy as np
import pytest
from onnx9000.backends.codegen.compiler import compile_cpp, load_pybind_module
from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor


def test_compile_add_graph():
    """Test end-to-end compilation of a graph."""
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

    so_path = compile_cpp("", use_pybind=True)  # dummy to get path
    import os

    mod_name = os.path.basename(so_path).split(".")[0]
    code = gen.generate(pybind_module_name=mod_name)
    so_path = compile_cpp(code, use_pybind=True, output_path=so_path)

    mod = load_pybind_module(so_path, module_name=mod_name)

    # Initialize the generated class
    # Since there are no initializers, the constructor takes no args.
    model = mod.GeneratedModel()

    A_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B_val = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    # forward() takes 2 arrays and returns 1
    C_val = model.forward(A_val, B_val)

    np.testing.assert_allclose(C_val, A_val + B_val)
