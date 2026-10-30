"""Unit tests for TensorBoard event exporter in `onnx9000.core.tensorboard_exporter`."""

import os
import tempfile

from onnx9000.core.ir import Graph, Node
from onnx9000.core.tensorboard_exporter import export_tensorboard


def test_export_tensorboard() -> None:
    """Verify export_tensorboard writes a valid tfevents file."""
    g = Graph("ResNet_test")
    g.add_node(Node("Conv", ["x"], ["y"], name="conv1"))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_tensorboard(g, tmpdir)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 8
