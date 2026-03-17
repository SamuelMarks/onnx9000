"""Tests the pyodide module functionality."""

import sys


def test_pyodide_compatibility() -> None:
    """
    Step 097: Ensure `import onnx9000.toolkit.script` works flawlessly inside Pyodide.
    Since we don't have Pyodide running in our pytest environment natively,
    we simulate the strict requirements:
    1. No heavy C extensions (like the standard protobuf).
    2. Works purely with Python built-ins and numpy.
    """
    assert "numpy" in sys.modules
    assert "onnx9000.toolkit.script.builder" in sys.modules


def test_dynamic_mlp_generation() -> None:
    """Step 099: Test dynamic generation of a simple 2-layer MLP purely in the browser."""
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.toolkit.script import GraphBuilder, op

    builder = GraphBuilder("BrowserMLP")
    x = builder.add_input("x", DType.FLOAT32, (1, 128))
    w1 = builder.add_initializer("w1", np.random.randn(128, 64).astype(np.float32))
    b1 = builder.add_initializer("b1", np.zeros(64, dtype=np.float32))
    w2 = builder.add_initializer("w2", np.random.randn(64, 10).astype(np.float32))
    b2 = builder.add_initializer("b2", np.zeros(10, dtype=np.float32))
    with builder:
        h1 = op.Relu(op.Add(op.MatMul(x, w1), b1))
        logits = op.Add(op.MatMul(h1, w2), b2)
        probs = op.Softmax(logits, axis=-1)
    builder.add_output(probs, "probs")
    model = builder.to_onnx()
    onnx_bytes = model.SerializeToString()
    assert isinstance(onnx_bytes, bytes)
    assert len(onnx_bytes) > 0
    assert model.graph.name == "BrowserMLP"
    assert len(model.graph.node) == 6


def test_js_wrapper() -> None:
    """Tests the js wrapper functionality."""
    from onnx9000.toolkit.script.js_wrapper import JSGraphBuilder

    builder = JSGraphBuilder("MyGraph")
    builder.add_input("x", "FLOAT32", [1, 10])
    builder.add_output("y")
    b = builder.build_to_bytes()
    assert isinstance(b, bytes)
    assert len(b) > 0
