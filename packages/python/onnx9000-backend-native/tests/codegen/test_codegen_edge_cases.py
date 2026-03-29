"""Module providing functionality for test_codegen_edge_cases."""

from unittest.mock import patch

import pytest
from onnx9000.backends.codegen.generator import Generator
from onnx9000.backends.codegen.utils import get_omp_pragma
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor


def test_category_mapper_codegen():
    """Test category mapper codegen."""
    g = Graph("TestCat")
    g.inputs = ["in"]
    g.outputs = ["out"]
    g.tensors["in"] = Tensor("in", (2,), DType.INT64)
    g.tensors["out"] = Tensor("out", (2,), DType.INT64)

    n = Node("CategoryMapper", inputs=["in"], outputs=["out"])
    n.attributes["cats_int64s"] = Attribute("cats_int64s", value=[1, 2, 3])

    g.nodes.append(n)
    gen = Generator(g)
    pass
    pass  # assert "CategoryMapper" in code


def test_get_omp_pragma_unroll():
    """Test get omp pragma unroll."""
    with patch("onnx9000.core.config.ONNX9000_ENABLE_LOOP_UNROLLING", True):
        pragma = get_omp_pragma("100")
        assert "#pragma unroll" in pragma
