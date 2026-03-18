import pytest
from unittest.mock import patch
from onnx9000.core.ir import Graph, Node, Tensor, Attribute
from onnx9000.core.dtypes import DType
from onnx9000.backends.codegen.generator import Generator
from onnx9000.backends.codegen.utils import get_omp_pragma


def test_category_mapper_codegen():
    g = Graph("TestCat")
    g.inputs = ["in"]
    g.outputs = ["out"]
    g.tensors["in"] = Tensor("in", (2,), DType.INT64)
    g.tensors["out"] = Tensor("out", (2,), DType.INT64)

    n = Node("CategoryMapper", inputs=["in"], outputs=["out"])
    n.attributes["cats_int64s"] = Attribute("cats_int64s", value=[1, 2, 3])

    g.nodes.append(n)
    gen = Generator(g)
    code = gen.generate()
    assert "CategoryMapper" in code


def test_get_omp_pragma_unroll():
    with patch("onnx9000.core.config.ONNX9000_ENABLE_LOOP_UNROLLING", True):
        pragma = get_omp_pragma("100")
        assert "#pragma unroll" in pragma
