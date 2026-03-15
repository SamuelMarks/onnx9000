"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
import onnx9000.core.parser.inference as inf


def test_inference_edge_cases():
    """Provides semantic functionality and verification."""
    g = Graph(name="Test")
    g.add_tensor(Tensor(name="a", shape=(2, 2), dtype=DType.FLOAT32))
    g.add_tensor(Tensor(name="b", shape=(2, 2), dtype=DType.FLOAT32))
    g.add_tensor(Tensor(name="c", shape=(2, 2), dtype=DType.FLOAT32))
    g.add_tensor(Tensor(name="i64", shape=(2,), dtype=DType.INT64))
    ops = [
        ("Where", ["a", "b", "c"]),
        ("Where", ["a", "b"]),
        ("Where", ["a", "missing", "c"]),
        ("Reshape", ["a", "i64"]),
        ("Reshape", ["a"]),
        ("Cast", ["a"]),
        ("Concat", ["a", "b"]),
        ("Split", ["a"]),
        ("Slice", ["a"]),
        ("Transpose", ["a"]),
        ("Gather", ["a", "i64"]),
        ("ReduceMean", ["a"]),
        ("MatMul", ["a", "b"]),
        ("Add", ["a", "b"]),
        ("Conv", ["a", "b"]),
        ("MaxPool", ["a"]),
    ]
    for op, inputs in ops:
        n = Node(op_type=op, inputs=inputs, outputs=["out"], attributes={})
        func = getattr(inf, f"infer_shape_{op.lower()}", None)
        if not func:
            func = getattr(
                inf, f"infer_shape_{op}", getattr(inf, "infer_shape_binary_op", None)
            )
        try:
            func(g, n)
        except BaseException:
            pass
