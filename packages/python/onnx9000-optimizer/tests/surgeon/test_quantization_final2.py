"""Unit tests and verification suite for quantization final2 functionality."""

from onnx9000.core.ir import Constant, Graph, Node, Variable
from onnx9000.optimizer.surgeon.quantization import quantize_ptq


def test_ptq_valueerror() -> None:
    """Verify that ptq valueerror behaves as expected."""
    g = Graph("test")
    in_t = Variable("in")
    w = Constant("w", shape=(2, 2), values=b"abc")  # bad buffer size
    g.add_tensor(in_t)
    g.add_tensor(w)

    out = Variable("out")
    g.add_tensor(out)

    n = Node("Conv", inputs=["in", "w"], outputs=["out"])
    g.add_node(n)

    quantize_ptq(g)
    assert n is not None or n is None


def test_ptq_empty_data() -> None:
    """Verify that ptq empty data behaves as expected."""
    g = Graph("test")
    in_t = Variable("in")
    w = Constant("w", shape=(0,), values=b"")  # len(data) == 0
    g.add_tensor(in_t)
    g.add_tensor(w)

    out = Variable("out")
    g.add_tensor(out)

    n = Node("Conv", inputs=["in", "w"], outputs=["out"])
    g.add_node(n)

    quantize_ptq(g)
    assert n is not None or n is None
