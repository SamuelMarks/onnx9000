import pytest
from unittest.mock import patch
from onnx9000.core.ir import Graph
from onnx9000.core.parser.passes import optimize


@patch("onnx9000.core.parser.passes.constant_folding")
@patch("onnx9000.core.parser.passes.fuse_consecutive_transpose")
@patch("onnx9000.core.parser.passes.fuse_matmul_add")
@patch("onnx9000.core.parser.memory.plan_memory")
def test_optimize_passes(
    mock_plan_memory, mock_fuse_matmul_add, mock_fuse_transpose, mock_const_fold
):
    g = Graph("test")
    optimize(g)
    mock_const_fold.assert_called_once_with(g)
    mock_fuse_transpose.assert_called_once_with(g)
    mock_fuse_matmul_add.assert_called_once_with(g)
    mock_plan_memory.assert_called_once_with(g)
