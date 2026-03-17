"""Tests the tf importer cov final2 module functionality."""

from onnx9000.converters.tf.importer import TFImporter


def test_tf_importer_unsupported_op() -> None:
    """Tests the tf importer unsupported op functionality."""
    tf_dict = {"node": [{"op": "UnknownOp", "name": "unk", "input": []}]}
    importer = TFImporter()
    graph = importer.parse(tf_dict)
    assert "unk" not in graph.nodes
