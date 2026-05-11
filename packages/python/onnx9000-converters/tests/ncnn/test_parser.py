"""Tests for NCNN parser."""

from onnx9000.converters.ncnn.parser import parse_param


def test_parse_param_basic():
    """Test parsing a basic NCNN param file."""
    content = """7767517
4 4
Input            data             0 1 data 0=224 1=224 2=3
Convolution      conv1            1 1 data conv1 0=64 1=7 2=1 3=2 4=3 5=1 6=3456
Pooling          pool1            1 1 conv1 pool1 0=0 1=3 2=2 3=1 4=0
ReLU             relu1            1 1 pool1 relu1
    """
    info = parse_param(content)
    assert info["magic"] == 7767517
    assert info["layer_count"] == 4
    assert info["blob_count"] == 4

    layers = info["layers"]
    assert len(layers) == 4

    assert layers[0]["type"] == "Input"
    assert layers[0]["name"] == "data"
    assert layers[0]["bottoms"] == []
    assert layers[0]["tops"] == ["data"]
    assert layers[0]["params"][0] == 224

    assert layers[1]["type"] == "Convolution"
    assert layers[1]["name"] == "conv1"
    assert layers[1]["bottoms"] == ["data"]
    assert layers[1]["tops"] == ["conv1"]
    assert layers[1]["params"][0] == 64

    assert layers[2]["type"] == "Pooling"
    assert layers[2]["bottoms"] == ["conv1"]
    assert layers[2]["tops"] == ["pool1"]

    assert layers[3]["type"] == "ReLU"


def test_parse_param_empty_and_invalid():
    """Test parsing invalid param files."""
    import pytest

    with pytest.raises(ValueError, match="Empty NCNN param file"):
        parse_param("   \n")

    with pytest.raises(ValueError, match="Invalid NCNN magic number"):
        parse_param("12345\n1 1\nInput data 0 1 data 0=224")


def test_parse_param_scientific():
    """Test parsing scientific notation."""
    content = """7767517
1 1
Input data 0 1 data 0=1.5e-3
    """
    info = parse_param(content)
    assert info["layers"][0]["params"][0] == 1.5e-3
