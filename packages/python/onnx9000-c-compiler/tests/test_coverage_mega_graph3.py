"""Tests for coverage mega graph3."""

import numpy as np
from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo


def test_compiler_rtos_and_wasm():
    """Docstring for D103."""
    g = Graph("RtosWasm")
    g.inputs.append(ValueInfo("X", (2,), DType.FLOAT32))
    g.outputs.append("Y")
    g.nodes.append(Node("Identity", ["X"], ["Y"]))
    c_rtos = C89Compiler(g, target="freertos")
    h, c = c_rtos.generate()
    assert "vTaskDelay" in c

    c_wasm = C89Compiler(g, target="wasm")
    h, c = c_wasm.generate()
    assert "ONNX9000_ALIGN_16" in c


def test_activations_swish_mish():
    """Docstring for D103."""
    g = Graph("SwishMish")
    g.inputs.append(ValueInfo("X", (2,), DType.FLOAT32))
    g.outputs.extend(["Y_swish", "Y_mish"])
    g.nodes.append(Node("Swish", ["X"], ["Y_swish"]))
    g.nodes.append(Node("Mish", ["X"], ["Y_mish"]))
    c_comp = C89Compiler(g)
    h, c = c_comp.generate()
    assert "Swish" in c or "Mish" in c or "mish" in c or "swish" in c
    # Test without math.h
    c_comp2 = C89Compiler(g, use_math_h=False)
    h2, c2 = c_comp2.generate()
    assert "FALLBACK_EXPF" in c2


def test_concat_complex():
    """Docstring for D103."""
    g = Graph("ConcatComplex")
    g.tensors["X1"] = Tensor("X1", (2, 3, 4), DType.FLOAT32)
    g.tensors["X2"] = Tensor("X2", (2, 3, 4), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (2, 6, 4), DType.FLOAT32)
    g.inputs.append(ValueInfo("X1", (2, 3, 4), DType.FLOAT32))
    g.inputs.append(ValueInfo("X2", (2, 3, 4), DType.FLOAT32))
    g.outputs.append("Y")

    class AttrMock:
        """Attr mock."""

        def __init__(self, val):
            """Init."""
            self.value = val

    g.nodes.append(Node("Concat", ["X1", "X2"], ["Y"], {"axis": AttrMock(1)}))
    c = C89Compiler(g)
    h, c_src = c.generate()
    assert "out_axis_offset" in c_src


def test_gru():
    """Docstring for D103."""
    g = Graph("GRU")
    g.tensors["X"] = Tensor("X", (2, 3, 4), DType.FLOAT32)
    g.tensors["W"] = Tensor("W", (1, 6, 4), DType.FLOAT32)
    g.tensors["R"] = Tensor("R", (1, 6, 2), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (2, 1, 3, 2), DType.FLOAT32)
    g.inputs.extend(
        [
            ValueInfo("X", (2, 3, 4), DType.FLOAT32),
            ValueInfo("W", (1, 6, 4), DType.FLOAT32),
            ValueInfo("R", (1, 6, 2), DType.FLOAT32),
        ]
    )
    g.outputs.append("Y")
    g.nodes.append(Node("GRU", ["X", "W", "R"], ["Y"]))
    c = C89Compiler(g)
    h, c_src = c.generate()
    assert "GRU Math" in c_src


def test_rnn():
    """Docstring for D103."""
    g = Graph("RNN")
    g.tensors["X"] = Tensor("X", (2, 3, 4), DType.FLOAT32)
    g.tensors["W"] = Tensor("W", (1, 6, 4), DType.FLOAT32)
    g.tensors["R"] = Tensor("R", (1, 6, 2), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (2, 1, 3, 2), DType.FLOAT32)
    g.inputs.extend(
        [
            ValueInfo("X", (2, 3, 4), DType.FLOAT32),
            ValueInfo("W", (1, 6, 4), DType.FLOAT32),
            ValueInfo("R", (1, 6, 2), DType.FLOAT32),
        ]
    )
    g.outputs.append("Y")
    g.nodes.append(Node("RNN", ["X", "W", "R"], ["Y"]))
    c = C89Compiler(g)
    h, c_src = c.generate()
    assert "Simple RNN Math" in c_src
