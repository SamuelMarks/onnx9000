import pytest
from pathlib import Path
import numpy as np
from onnx9000.export.builder import (
    build_graph_proto,
    build_model_proto,
    validate_model,
    sanitize_model,
    to_string,
    to_onnx,
)
from onnx9000.frontends.frontend.builder import GraphBuilder
from onnx9000.frontends.frontend.tensor import Tensor, Parameter, Node
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import CompilationError
from onnx9000.core import onnx_pb2


def test_builder_coverage(tmp_path):
    gb = GraphBuilder("test_graph")
    t_in = Tensor(name="input1", shape=(1,), dtype=DType.FLOAT32)
    gb.inputs.append(t_in)
    t_out = Tensor(name="output1", shape=(1,), dtype=DType.FLOAT32)
    gb.outputs.append(t_out)
    p = Parameter(name="param1", data=np.array([1.0], dtype=np.float32))
    t_data = Tensor(name="t_data", shape=(1,), dtype=DType.FLOAT32)
    t_data.data = np.array([2.0], dtype=np.float32)
    n1 = Node(
        op_type="Relu",
        inputs=[t_in, p, t_data],
        outputs=[t_out, "unnamed_out"],
        attributes={},
    )
    gb.nodes.append(n1)
    graph_proto = build_graph_proto(gb)
    assert graph_proto.name == "test_graph"
    model_proto = build_model_proto(gb)
    assert model_proto.ir_version == 8
    validate_model(model_proto)
    bad_model = onnx_pb2.ModelProto()
    with pytest.raises(CompilationError, match="valid graph"):
        validate_model(bad_model)
    b = to_string(gb)
    assert len(b) > 0
    out_file = tmp_path / "model.onnx"
    to_onnx(gb, out_file)
    assert out_file.exists()


def test_builder_parameter():
    gb = GraphBuilder("test_param")
    p2 = Parameter(name="param2", data=np.array([2.0], dtype=np.float32))
    t_out = Tensor(name="output2", shape=(1,), dtype=DType.FLOAT32)
    gb.outputs.append(t_out)
    n2 = Node(op_type="Abs", inputs=[p2], outputs=[t_out], attributes={})
    gb.nodes.append(n2)
    t_intermediate = Tensor(name="inter", shape=(1,), dtype=DType.FLOAT32)
    gb.nodes.append(
        Node(op_type="Abs", inputs=[t_out], outputs=[t_intermediate], attributes={})
    )
    build_graph_proto(gb)


def test_builder_parameter_no_data():
    gb = GraphBuilder("test_param_nodata")
    p3 = Parameter(name="param_nodata", data=None, shape=(1,), dtype=DType.FLOAT32)
    t_out = Tensor(name="out_nodata", shape=(1,), dtype=DType.FLOAT32)
    gb.outputs.append(t_out)
    n3 = Node(op_type="Abs", inputs=[p3], outputs=[t_out], attributes={})
    gb.nodes.append(n3)
    build_graph_proto(gb)
