"""Tests for MXNet parser."""

import json

from onnx9000.converters.mxnet.parser import parse_symbol


def test_parse_symbol_basic():
    """Test parsing MXNet symbol JSON."""
    content = json.dumps({"nodes": [{"op": "null", "name": "data"}], "heads": [[0, 0, 0]]})
    symbol_info = parse_symbol(content)
    assert len(symbol_info["nodes"]) == 1
    assert len(symbol_info["heads"]) == 1
