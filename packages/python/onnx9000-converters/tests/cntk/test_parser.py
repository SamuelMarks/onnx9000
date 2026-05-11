"""Tests for CNTK parser."""

from onnx9000.converters.cntk.parser import parse_cntk_model


def test_parse_cntk_basic():
    """Test parsing CNTK model bytes."""
    # Dummy protobuf
    data = b"\x0a\x04test"
    model_dict = parse_cntk_model(data)
    assert "nodes" in model_dict
    assert "inputs" in model_dict
    assert "outputs" in model_dict
