import pytest
from onnx9000.frontends.tf.importer import TFImporter, load_tf


def test_importer_unknown_op():
    graph_def = {"node": [{"op": "UnknownOp", "name": "unknown_node"}]}
    importer = TFImporter()
    g = importer.parse(graph_def)
    assert len(g.nodes) == 0


def test_load_tf_not_implemented():
    with pytest.raises(
        NotImplementedError, match="Parsing physical file not implemented in mock"
    ):
        load_tf("fake_path.pb")
