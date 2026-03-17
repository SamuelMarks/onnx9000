"""Tests the builder cov final module functionality."""

import numpy as np
from onnx9000.core.onnx_pb2 import AttributeProto, GraphProto, NodeProto, TensorProto
from onnx9000.toolkit.script.builder import GraphBuilder


def test_builder_from_onnx_cov() -> None:
    """Tests the builder from onnx cov functionality."""
    g_proto = GraphProto(name="test_graph")
    init = TensorProto(name="int_init", data_type=7, dims=[2])
    init.raw_data = np.array([1, 2], dtype=np.int64).tobytes()
    g_proto.initializer.append(init)
    node = NodeProto(op_type="If", name="if_node")
    attr = AttributeProto(name="then_branch", type=5)
    attr.g.CopyFrom(GraphProto(name="sub_graph"))
    node.attribute.append(attr)
    g_proto.node.append(node)
    GraphBuilder.from_onnx(g_proto)
