"""Integration test for MXNet converter."""

import json
import os
import tempfile

from onnx9000.converters.mxnet import MXNetConverter


def test_mxnet_snapshot_integration():
    """Verify MXNet snapshot exists."""
    snapshot_path = os.path.join(
        os.path.dirname(__file__), "../../../../../snapshots/mxnet-1.9.1.json"
    )
    assert os.path.exists(snapshot_path)
    with open(snapshot_path) as f:
        data = json.load(f)
    assert "version" in data
    assert "objects" in data


def test_mxnet_converter_parse():
    """Test full parse pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "model.params")
        with open(weights_path, "wb") as f:
            f.write(b"")  # mock empty params

        converter = MXNetConverter(weights_path)

        symbol_path = os.path.join(tmpdir, "model-symbol.json")
        with open(symbol_path, "w") as f:
            json.dump({"nodes": [{"op": "null", "name": "data"}], "heads": [[0, 0, 0]]}, f)

        graph = converter.parse(symbol_path)
        assert graph.name == "MXNetModel"

        # Test string parsing
        graph2 = converter.parse(
            '{"nodes": [{"op": "null", "name": "data"}], "heads": [[0, 0, 0]]}'
        )
        assert graph2.name == "MXNetModel"
