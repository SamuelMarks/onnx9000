import inspect
import pytest
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.core import dtypes


def test_onnx_importer_all():
    from onnx9000.tvm.relay.frontend.onnx import ONNXImporter

    importer = ONNXImporter()

    # call all _convert_* methods
    for name in dir(importer):
        if name.startswith("_convert_") and name != "_convert_map":
            method = getattr(importer, name)
            method([], {})

    # mock from_onnx
    import struct

    init_data = struct.pack("<f", 1.0)
    init_tensor = Tensor(name="init", dtype="float32", shape=(1,), data=init_data)

    in_vi = ValueInfo(name="in", dtype="float32", shape=(1,))
    out_vi = ValueInfo(name="out3", dtype="float32", shape=(1,))

    graph = Graph(name="test")
    graph.inputs = [in_vi]
    graph.outputs = [out_vi]
    graph.nodes = [
        Node(op_type="Add", inputs=["in", "init"], outputs=["out"]),
        Node(op_type="Unknown", inputs=["out"], outputs=["out1", "out2"]),
        Node(op_type="Relu", inputs=["out1"], outputs=["out3"]),
    ]
    graph.initializers = [init_tensor]

    importer.from_onnx(graph)

    # graph with multiple outputs
    graph2 = Graph(name="test2")
    graph2.inputs = [
        ValueInfo(name="in1", dtype="float32", shape=("dim",)),
        ValueInfo(name="in2", dtype="float32", shape=(1,)),
    ]
    graph2.outputs = [
        ValueInfo(name="in1", dtype="float32", shape=("dim",)),
        ValueInfo(name="in2", dtype="float32", shape=(1,)),
    ]
    graph2.nodes = []
    graph2.initializers = []

    importer.from_onnx(graph2)
