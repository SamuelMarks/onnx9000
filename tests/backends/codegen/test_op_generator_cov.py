import importlib
import pytest
from unittest.mock import patch
from onnx9000.backends.codegen.op_generator import OpGenerator
from onnx9000.core.ir import Node


class DummyOpGen(OpGenerator):
    def generate(self, node: Node, generator_context) -> str:
        return super().generate(node, generator_context)


def test_op_generator():
    g = DummyOpGen()
    assert g.generate(Node("dummy", [], [], {}), None) is None


def test_type_checking():
    with patch("typing.TYPE_CHECKING", True):
        import onnx9000.backends.codegen.op_generator as op_generator

        importlib.reload(op_generator)
    importlib.reload(op_generator)


def test_nn_empty_output():
    from onnx9000.backends.codegen.ops.nn import generate_attention
    from onnx9000.core.ir import Node, Graph, Tensor
    from onnx9000.backends.codegen.generator import Generator
    from onnx9000.core.dtypes import DType

    node = Node("Attention", ["a"], ["", "c"], {})
    graph = Graph("test")
    t_c = Tensor("c", (1,), DType.FLOAT32)
    t_c.buffer_id = 1
    graph.tensors["c"] = t_c
    ctx = Generator(graph)
    ctx.get_tensor_name = lambda x: x
    res = generate_attention(node, ctx)
    assert "// Attention (Mock)" in res
