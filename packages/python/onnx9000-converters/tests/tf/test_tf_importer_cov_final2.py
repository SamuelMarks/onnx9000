import onnx9000.converters.tf.parsers as parsers


def test_missing_tf_fns():
    if hasattr(parsers, "extract_variables"):
        assert parsers.extract_variables("dir") == {"dir": b"0000"}
    if hasattr(parsers, "convert_tf_shape"):
        assert parsers.convert_tf_shape([1, 2, -1, 0]) == [1, 2, -1, -1]

    class DummyNode:
        def __init__(self):
            self.op = "MyOp"
            self.name = "MyName"

    n = DummyNode()
    if hasattr(parsers, "log_unsupported_node"):
        parsers.log_unsupported_node(n)

    if hasattr(parsers, "create_custom_node"):
        parsers.create_custom_node(n)
        assert n.op == "Custom_MyOp"
