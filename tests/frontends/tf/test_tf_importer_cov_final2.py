from onnx9000.frontends.tf.importer import TFImporter
import pytest


def test_tf_importer_unsupported_op():
    tf_dict = {"node": [{"op": "UnknownOp", "name": "unk", "input": []}]}
    importer = TFImporter()
    graph = importer.parse(tf_dict)
    assert "unk" not in graph.nodes
