"""Tests the importer cov final module functionality."""

from onnx9000.converters.tf.importer import TFImporter, load_tf


def test_importer_unknown_op() -> None:
    """Tests the importer unknown op functionality."""
    graph_def = {"node": [{"op": "UnknownOp", "name": "unknown_node"}]}
    importer = TFImporter()
    g = importer.parse(graph_def)
    assert len(g.nodes) == 0


def test_load_tf_not_implemented() -> None:
    """Tests the load tf not implemented functionality."""
    g = load_tf("fake_path.pb")
    assert len(g.nodes) == 0
