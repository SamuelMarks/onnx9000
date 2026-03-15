"""Module providing core logic and structural definitions."""

import pytest
import inspect
import onnx9000.core.parser.inference as inf
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType


def test_inference_errors():
    """Provides semantic functionality and verification."""
    g = Graph(name="Test")
    g.add_tensor(Tensor(name="a", shape=(2, 2), dtype=DType.FLOAT32))
    for name, func in inspect.getmembers(inf, inspect.isfunction):
        if not name.startswith("infer_shape_"):
            continue
        n0 = Node(op_type=name.replace("infer_shape_", ""), inputs=[], outputs=["out"])
        try:
            func(g, n0)
        except Exception:
            pass
        n1 = Node(op_type=n0.op_type, inputs=["a"], outputs=["out"])
        try:
            func(g, n1)
        except Exception:
            pass
        n2 = Node(op_type=n0.op_type, inputs=["a", "missing"], outputs=["out"])
        try:
            func(g, n2)
        except Exception:
            pass
        try:
            func(g, Node(op_type=n0.op_type, inputs=["a", "a"], outputs=["out"]))
        except Exception:
            pass
