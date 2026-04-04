"""Module docstring."""

import pytest
from onnx9000.core.ir import DType, Graph, Node, ValueInfo, Variable
from onnx9000.optimizer.surgeon.obfuscator import obfuscate_names


def test_obfuscate_names_with_interface():
    """Docstring for D103."""
    graph = Graph("test_graph")

    # Input and Output
    graph.inputs = [ValueInfo("IN", [1, 10], DType.FLOAT32)]
    graph.outputs = [ValueInfo("OUT", [1, 10], DType.FLOAT32)]

    t1 = Variable("IN", [1, 10], DType.FLOAT32)
    t2 = Variable("INTERNAL", [1, 10], DType.FLOAT32)
    t3 = Variable("OUT", [1, 10], DType.FLOAT32)

    graph.tensors = {"IN": t1, "INTERNAL": t2, "OUT": t3}

    n1 = Node("Identity", ["IN"], ["INTERNAL"], {}, "id1")
    n2 = Node("Identity", ["INTERNAL"], ["OUT"], {}, "id2")
    graph.nodes = [n1, n2]

    obfuscated = obfuscate_names(graph)

    # IN and OUT should remain
    assert "IN" in obfuscated.tensors
    assert "OUT" in obfuscated.tensors
    # INTERNAL should be renamed
    assert "INTERNAL" not in obfuscated.tensors
    assert any(name.startswith("t_") for name in obfuscated.tensors if name not in ["IN", "OUT"])
