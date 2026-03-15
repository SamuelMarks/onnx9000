"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.core.parser.inference import _INFERENCE_RULES


def test_trilu():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(Tensor("mat", (4, 4), DType.FLOAT32))
    node = Node("Trilu", inputs=["mat"], outputs=["out"], attributes={})
    _INFERENCE_RULES["Trilu"](node, g)
    assert g.tensors["out"].shape == (4, 4)
