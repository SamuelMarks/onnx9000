"""Tests for Torch FX parser coverage gaps."""

from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
    assert parser._get_shape([1, 2, 3]) == (1, 2, 3)
    assert parser._get_shape(torch.Size([1, 2, 3])) == (1, 2, 3)
