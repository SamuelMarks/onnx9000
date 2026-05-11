"""Integration test for CNTK converter."""

import json
import os
import tempfile

from onnx9000.converters.cntk import CNTKConverter


def test_cntk_snapshot_integration():
    """Verify CNTK snapshot exists."""
    snapshot_path = os.path.join(
        os.path.dirname(__file__), "../../../../../snapshots/cntk-2.7.post2.json"
    )
    assert os.path.exists(snapshot_path)
    with open(snapshot_path) as f:
        data = json.load(f)
    assert "version" in data
    assert "objects" in data


def test_cntk_converter_parse():
    """Test full parse pipeline."""
    converter = CNTKConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test.model")
        with open(model_path, "wb") as f:
            f.write(b"")

        graph = converter.parse(model_path)
        assert graph.name == "CNTKModel"

    # Also string/bytes
    graph2 = converter.parse(b"")
    assert graph2.name == "CNTKModel"
