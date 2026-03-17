"""Tests the session api extras module functionality."""

import numpy as np
import pytest
from onnx9000.backends.cpu.executor import CPUExecutionProvider
from onnx9000.backends.session import InferenceSession
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.serializer import to_bytes


def test_session_bytes_and_empty() -> None:
    """Tests the session bytes and empty functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("T", (1,), DType.FLOAT32, is_initializer=True))
    g.initializers.append("T")
    b = to_bytes(g)
    s = InferenceSession(b, providers=None)
    res = s.get_overridable_initializers()
    assert len(res) == 1
    assert res[0].name == "T"
    assert s.get_provider_options() == {}
    p = CPUExecutionProvider({})
    s.set_providers([p])
    assert s.get_providers() == ["CPUExecutionProvider"]
    assert s.get_provider_options() == {"CPUExecutionProvider": {}}
    s.providers = []
    s.graph.add_node(Node("Add", ["T", "T"], ["Out"], attributes={}))
    s.run(None, {})


def test_session_missing_output() -> None:
    """Tests the session missing output functionality."""
    from onnx9000.backends.session import InferenceSessionError

    g = Graph("g")
    g.add_node(Node("Add", ["A", "A"], ["C"], attributes={}))
    s = InferenceSession(g, providers=[CPUExecutionProvider({})])
    with pytest.raises(InferenceSessionError, match="Requested output MissingOut was not computed"):
        s.run(
            ["MissingOut"],
            {
                "A": Tensor(
                    "A", (1,), DType.FLOAT32, data=np.array([1.0], dtype=np.float32).tobytes()
                )
            },
        )
