from onnx9000.frontend.tf.importer import TFImporter


def test_tf_importer_unsupported_op() -> None:
    tf_dict = {"node": [{"op": "UnknownOp", "name": "unk", "input": []}]}
    importer = TFImporter()
    graph = importer.parse(tf_dict)
    assert "unk" not in graph.nodes
