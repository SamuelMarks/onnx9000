import pytest
import inspect


def test_onnx_importer_all():
    from onnx9000.tvm.relay.frontend.onnx import ONNXImporter
    import onnx
    from onnx import helper
    from onnx import TensorProto
    from onnx import AttributeProto

    importer = ONNXImporter()

    # cover _get_type
    t1 = helper.make_tensor_value_info("test", TensorProto.FLOAT, [1, "dim", None])
    ty = importer._get_type(t1.type.tensor_type)

    # cover _parse_attr
    a_f = helper.make_attribute("a", 1.0)
    assert importer._parse_attr(a_f) == 1.0
    a_i = helper.make_attribute("a", 1)
    assert importer._parse_attr(a_i) == 1
    a_s = helper.make_attribute("a", b"hello")
    assert importer._parse_attr(a_s) == "hello"

    import numpy as np

    a_t = helper.make_attribute("a", helper.make_tensor("t", TensorProto.FLOAT, [1], [1.0]))
    importer._parse_attr(a_t)

    a_fs = helper.make_attribute("a", [1.0, 2.0])
    importer._parse_attr(a_fs)
    a_is = helper.make_attribute("a", [1, 2])
    importer._parse_attr(a_is)
    a_ss = helper.make_attribute("a", [b"a", b"b"])
    importer._parse_attr(a_ss)

    class UnkAttr:
        type = 999

    importer._parse_attr(UnkAttr())

    # call all _convert_* methods
    for name in dir(importer):
        if name.startswith("_convert_") and name != "_convert_map":
            method = getattr(importer, name)
            method([], {})

    # mock from_onnx
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Add", ["in", "init"], ["out"]),
            helper.make_node("Unknown", ["out"], ["out1", "out2"]),
            helper.make_node("Relu", ["out1"], ["out3"]),
        ],
        name="test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("out3", TensorProto.FLOAT, [1])],
        initializer=[helper.make_tensor("init", TensorProto.FLOAT, [1], [1.0])],
    )
    model = helper.make_model(graph)
    importer.from_onnx(model)

    # graph with multiple outputs
    graph2 = helper.make_graph(
        nodes=[],
        name="test",
        inputs=[
            helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("in2", TensorProto.FLOAT, [1]),
        ],
        outputs=[
            helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("in2", TensorProto.FLOAT, [1]),
        ],
        initializer=[],
    )
    importer.from_onnx(helper.make_model(graph2))
