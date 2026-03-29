"""Tests for a large graph with many operations in the C compiler."""

import sys

import pytest
from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo


def test_mega_graph_routing_coverage():
    """Test routing coverage for a mega graph with various operations."""
    tensors = {
        "X": Tensor("X", [1, 2, 2], DType.FLOAT32, data=b"\x00" * 16),
        "Y": Tensor("Y", [1, 2, 2], DType.FLOAT32, data=b"\x00" * 16),
        "Z": Tensor("Z", [1, 2, 2], DType.FLOAT32, data=b"\x00" * 16),
    }
    for i in range(1, 24):
        tensors[f"O{i}"] = Tensor(f"O{i}", [1, 2, 2], DType.FLOAT32, data=b"\x00" * 16)
    # Gather out: [1, 1, 2, 2, 2, 2] (wait, axis 0: indices_shape + in_shape[1:])
    # For X[1, 2, 2], Y[1, 2, 2], axis 0 => out_shape is Y.shape + X.shape[1:] = [1, 2, 2] + [2, 2] = [1, 2, 2, 2, 2]
    tensors["O1"].shape = [1, 2, 2, 2, 2]

    # ScatterElements out: same as input X
    tensors["O2"].shape = [1, 2, 2]

    # ScatterND out: same as input X
    tensors["O3"].shape = [1, 2, 2]

    # Expand: [1, 2, 2]
    tensors["O4"].shape = [1, 2, 2]

    # Tile: X[1, 2, 2], Y is repeats, let's say Y is [2, 2, 2]. Output is [2, 4, 4]
    tensors["O5"].shape = [2, 4, 4]

    # GatherND: indices Y [1, 2, 2], X [1, 2, 2]. indices_shape[:-1] + X.shape[indices_shape[-1]:]
    # Y[-1] is 2. X.shape[2:] is [2]. So [1, 2] + [2] = [1, 2, 2]
    tensors["O6"].shape = [1, 2, 2]

    # DepthToSpace/SpaceToDepth: shape transformations
    tensors["O10"].shape = [1, 8, 1, 1]  # DepthToSpace blocksize 2
    tensors["O11"].shape = [1, 2, 4, 4]  # SpaceToDepth blocksize 2

    # Split
    tensors["O22"].shape = [1, 1, 2]
    tensors["O23"].shape = [1, 1, 2]

    inputs = [ValueInfo("X", [1, 2, 2], DType.FLOAT32)]
    outputs = [ValueInfo("O", [1, 2, 2], DType.FLOAT32)]

    nodes = [
        Node("Gather", ["X", "Y"], ["O1"]),
        Node("ScatterElements", ["X", "Y", "Z"], ["O2"]),
        Node("ScatterND", ["X", "Y", "Z"], ["O3"]),
        Node("Expand", ["X", "Y"], ["O4"]),
        Node("Tile", ["X", "Y"], ["O5"]),
        Node("GatherND", ["X", "Y"], ["O6"]),
        Node("CumSum", ["X", "Y"], ["O7"]),
        Node("ReverseSequence", ["X", "Y"], ["O8"]),
        Node("OneHot", ["X", "Y", "Z"], ["O9"]),
        Node("DepthToSpace", ["X"], ["O10"], attributes={"blocksize": 2}),
        Node("SpaceToDepth", ["X"], ["O11"], attributes={"blocksize": 2}),
        Node("ConstantOfShape", ["X"], ["O12"]),
        Node("Slice", ["X", "Y", "Z"], ["O13"]),
        Node("Less", ["X", "Y"], ["O14"]),
        Node("LessOrEqual", ["X", "Y"], ["O15"]),
        Node("Greater", ["X", "Y"], ["O16"]),
        Node("GreaterOrEqual", ["X", "Y"], ["O17"]),
        Node("And", ["X", "Y"], ["O18"]),
        Node("Or", ["X", "Y"], ["O19"]),
        Node("Xor", ["X", "Y"], ["O20"]),
        Node("Not", ["X"], ["O21"]),
        Node("Split", ["X"], ["O22", "O23"]),
    ]

    graph = Graph("Mega")
    graph.nodes = nodes
    graph.inputs = inputs
    graph.outputs = outputs
    graph.tensors = tensors
    from onnx9000.core.shape_inference import infer_shapes_and_types

    try:
        infer_shapes_and_types(graph)
    except Exception as e:
        print("Infer failed:", e)

    compiler = C89Compiler(graph=graph)
    _, c_code = compiler.generate()
    assert "ScatterElements" in c_code
    assert "CumSum" in c_code
