"""Integration tests for Darknet parser."""

import os
import struct
import tempfile

import numpy as np
from onnx9000.converters.darknet import DarknetConverter
from onnx9000.converters.darknet.weights import load_weights


def test_integration_full():
    """Test full integration."""
    cfg_content = """
    [net]
    batch=1
    subdivisions=1
    width=416
    height=416
    channels=3
    momentum=0.9
    decay=0.0005

    [convolutional]
    batch_normalize=1
    filters=16
    size=3
    stride=1
    pad=1
    activation=leaky
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "model.cfg")
        with open(cfg_path, "w") as f:
            f.write(cfg_content)

        weights_path = os.path.join(tmpdir, "model.weights")
        with open(weights_path, "wb") as f:
            # Header: major=0, minor=2, revision=0, seen=0 (64-bit)
            f.write(struct.pack("iii", 0, 2, 0))
            f.write(struct.pack("q", 0))

            # weights for bn (16 * 4) + conv (16 * 3 * 3 * 3 = 432)
            # Total: 64 + 432 = 496
            f.write(np.zeros(496, dtype=np.float32).tobytes())

        converter = DarknetConverter(weights_path)
        graph = converter.parse(cfg_path)

        assert len(graph.nodes) == 3
        assert graph.nodes[0].op_type == "Conv"
        assert graph.nodes[1].op_type == "BatchNormalization"
        assert graph.nodes[2].op_type == "LeakyRelu"

        # Also test loading from string
        graph2 = converter.parse(cfg_content)
        assert len(graph2.nodes) == 3


def test_load_weights_32bit():
    """Test loading weights with 32-bit seen count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "model.weights")
        with open(weights_path, "wb") as f:
            # Header: major=0, minor=1, revision=0, seen=0 (32-bit)
            f.write(struct.pack("iii", 0, 1, 0))
            f.write(struct.pack("i", 0))
            f.write(np.zeros(10, dtype=np.float32).tobytes())

        with open(weights_path, "rb") as f:
            data = load_weights(f)
            assert data["major"] == 0
            assert data["minor"] == 1
            assert data["seen"] == 0
            assert len(data["weights"]) == 10


def test_load_weights_invalid():
    """Test loading invalid weights."""
    import pytest

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "model.weights")

        with open(weights_path, "wb") as f:
            f.write(b"123")

        with open(weights_path, "rb") as f:
            with pytest.raises(ValueError, match="too short for header"):
                load_weights(f)

        with open(weights_path, "wb") as f:
            f.write(struct.pack("iii", 0, 2, 0))
            f.write(b"123")  # short seen

        with open(weights_path, "rb") as f:
            with pytest.raises(ValueError, match="too short for seen"):
                load_weights(f)

        with open(weights_path, "wb") as f:
            f.write(struct.pack("iii", 0, 1, 0))
            f.write(b"123")  # short seen

        with open(weights_path, "rb") as f:
            with pytest.raises(ValueError, match="too short for seen"):
                load_weights(f)
