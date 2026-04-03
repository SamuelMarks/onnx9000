"""Tests for Torch FX parser coverage gaps."""

from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from onnx9000.converters.torch.export import ExportParser
from onnx9000.converters.torch.fx import FXParser


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
def test_fx_parser_basic():
    """Test basic Torch FX parser."""

    class SimpleModule(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    m = SimpleModule()
    # torch.fx is available in modern torch
    gm = torch.fx.symbolic_trace(m)

    parser = FXParser(gm)
    graph = parser.parse()
    assert len(graph.nodes) > 0
    assert any(n.op_type == "Add" for n in graph.nodes)


def test_fx_parser_dtype_mapping():
    """Test FX parser dtype mapping."""
    if not TORCH_AVAILABLE:
        return
    mock_gm = MagicMock()
    parser = FXParser(mock_gm)
    from onnx9000.core.dtypes import DType

    assert parser._get_dtype(torch.float32) == DType.FLOAT32
    assert parser._get_dtype(torch.int32) == DType.INT32


def test_fx_parser_shape_mapping():
    """Test FX parser shape mapping."""
    if not TORCH_AVAILABLE:
        return
    mock_gm = MagicMock()
    parser = FXParser(mock_gm)
    assert parser._get_shape(None) == ()
    assert parser._get_shape([1, 2, 3]) == (1, 2, 3)
    assert parser._get_shape(torch.Size([1, 2, 3])) == (1, 2, 3)
    assert parser._get_shape(["sym", 2]) == ("sym", 2)  # hit string branch


def test_fx_parser_nodes():
    """Test FX parser nodes."""
    if not TORCH_AVAILABLE:
        return

    mock_gm = MagicMock()
    # mock named_buffers for get_attr
    mock_gm.named_buffers.return_value = [("my_attr", None)]
    # mock getattr
    mock_attr = MagicMock(spec=torch.Tensor)
    mock_attr.shape = [2, 2]
    mock_attr.dtype = torch.float32
    mock_attr.detach().cpu().numpy.return_value = np.array([[1, 2], [3, 4]])
    mock_gm.my_attr = mock_attr

    parser = FXParser(mock_gm)

    # 1. get_attr
    mock_node_attr = MagicMock()
    mock_node_attr.op = "get_attr"
    mock_node_attr.target = "my_attr"
    mock_node_attr.name = "my_attr_node"
    parser._parse_node(mock_node_attr)
    assert len(parser.builder.parameters) == 1

    # 2. call_method
    mock_node = MagicMock(spec=torch.fx.Node)
    mock_node.op = "call_method"
    mock_node.target = "mul"
    mock_node.args = [42.0]
    mock_node.kwargs = {}
    mock_node.name = "mul_node"
    mock_node.meta = {"nn_module_stack": "stack"}
    parser._parse_node(mock_node)
    assert parser.builder.nodes[0].op_type == "Mul"

    # 3. call_module
    mock_node2 = MagicMock(spec=torch.fx.Node)
    mock_node2.op = "call_module"
    mock_node2.target = "relu"
    mock_node2.args = []
    mock_node2.kwargs = {}
    mock_node2.name = "relu_node"
    mock_node2.meta = {}
    mock_sub = MagicMock()
    mock_sub.__class__.__name__ = "relu"
    mock_gm.get_submodule.return_value = mock_sub
    parser._parse_node(mock_node2)
    assert parser.builder.nodes[1].op_type == "Relu"

    # 4. output
    parser.node_map[mock_node] = MagicMock()
    parser.node_map[mock_node2] = MagicMock()
    mock_node_out = MagicMock()
    mock_node_out.op = "output"
    mock_node_out.args = [(mock_node, mock_node2)]
    parser._parse_node(mock_node_out)
    assert len(parser.builder.outputs) == 2


def test_export_parser():
    """Test ExportParser."""
    if not TORCH_AVAILABLE:
        return
    mock_ep = MagicMock()
    mock_ep.graph_module = MagicMock()

    parser = ExportParser(mock_ep)
    parser.parse()
