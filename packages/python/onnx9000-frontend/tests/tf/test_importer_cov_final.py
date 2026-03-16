from onnx9000.frontend.tf.importer import TFImporter, load_tf


def test_importer_unknown_op() -> None:
    graph_def = {"node": [{"op": "UnknownOp", "name": "unknown_node"}]}
    importer = TFImporter()
    g = importer.parse(graph_def)
    assert len(g.nodes) == 0


def test_load_tf_not_implemented() -> None:
    g = load_tf("fake_path.pb")
    assert len(g.nodes) == 0
