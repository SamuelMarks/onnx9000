"""Integration tests for NCNN parser."""

import os
import struct
import tempfile

import numpy as np
from onnx9000.converters.ncnn import NCNNConverter
from onnx9000.converters.ncnn.weights import WeightsReader


def test_integration_full():
    """Test full integration."""
    param_content = """7767517
2 2
Input            data             0 1 data 0=224 1=224 2=3
Convolution      conv1            1 1 data conv1 0=16 1=3 5=1 6=432
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        param_path = os.path.join(tmpdir, "model.param")
        with open(param_path, "w") as f:
            f.write(param_content)

        bin_path = os.path.join(tmpdir, "model.bin")
        with open(bin_path, "wb") as f:
            # 16 biases -> 16 * 4 = 64 bytes (assuming float32 tag)
            f.write(struct.pack("I", 0))  # No magic / f32
            f.write(np.zeros(431, dtype=np.float32).tobytes())

            f.write(struct.pack("I", 0))  # No magic / f32
            f.write(np.zeros(15, dtype=np.float32).tobytes())

        converter = NCNNConverter(bin_path)
        graph = converter.parse(param_path)

        assert len(graph.nodes) == 1
        assert graph.nodes[0].op_type == "Conv"
        assert graph.nodes[0].name == "conv1"
        assert len(graph.nodes[0].inputs) == 3

        # string loading
        graph2 = converter.parse(param_content)
        assert len(graph2.nodes) == 1


def test_weights_reader_float16():
    """Test reading float16 weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "model.bin")
        with open(bin_path, "wb") as f:
            f.write(struct.pack("I", 0x01306B47))  # f16
            f.write(np.zeros(10, dtype=np.float16).tobytes())

        with open(bin_path, "rb") as f:
            reader = WeightsReader(f)
            data = reader.read_blob(10)
            assert len(data) == 10
            assert data.dtype == np.float32


def test_weights_reader_int8():
    """Test reading int8 weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "model.bin")
        with open(bin_path, "wb") as f:
            f.write(struct.pack("I", 0x000D4B38))  # i8
            f.write(np.zeros(10, dtype=np.int8).tobytes())

        with open(bin_path, "rb") as f:
            reader = WeightsReader(f)
            data = reader.read_blob(10)
            assert len(data) == 10
            assert data.dtype == np.float32
