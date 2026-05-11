"""Tests for Darknet parser."""

from onnx9000.converters.darknet.parser import parse_cfg


def test_parse_cfg_basic():
    """Test parsing a basic cfg file."""
    cfg = """
    [net]
    batch=1
    subdivisions=1
    width=416
    height=416
    channels=3
    momentum=0.9
    decay=0.0005
    angle=0
    saturation = 1.5
    exposure = 1.5
    hue=.1

    [convolutional]
    batch_normalize=1
    filters=32
    size=3
    stride=1
    pad=1
    activation=leaky

    [route]
    layers = -1, 8
    """

    blocks = parse_cfg(cfg)
    assert len(blocks) == 3
    assert blocks[0]["type"] == "net"
    assert blocks[0]["batch"] == "1"
    assert blocks[1]["type"] == "convolutional"
    assert blocks[1]["filters"] == "32"
    assert blocks[2]["type"] == "route"
    assert blocks[2]["layers"] == "-1, 8"


def test_parse_cfg_empty_and_comments():
    """Test parsing with comments and empty lines."""
    cfg = """
    # This is a comment
    
    [net]
    # Another comment
    batch=1
    
    """
    blocks = parse_cfg(cfg)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "net"
    assert blocks[0]["batch"] == "1"
