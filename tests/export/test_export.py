"""Module docstring."""

from pathlib import Path

import onnx9000
from onnx9000 import onnx_pb2
from onnx9000.dtypes import DType


def test_export_add(temp_dir: Path):
    """test_export_add docstring."""

    @onnx9000.jit
    def add_model(x, y):
        """add_model docstring."""

        return x + y

    x = onnx9000.Tensor(shape=(10,), dtype=DType.FLOAT32, name="x")
    y = onnx9000.Tensor(shape=(10,), dtype=DType.FLOAT32, name="y")

    builder = add_model(x, y)

    out_path = temp_dir / "add.onnx"
    onnx9000.to_onnx(builder, out_path)

    assert out_path.exists()

    # Reload and check basic proto fields manually
    model_proto = onnx_pb2.ModelProto()
    with open(out_path, "rb") as f:
        model_proto.ParseFromString(f.read())

    assert model_proto.producer_name == "onnx9000"
    assert len(model_proto.graph.node) == 1
    assert model_proto.graph.node[0].op_type == "Add"


def test_export_matmul_with_params(temp_dir: Path):
    """test_export_matmul_with_params docstring."""

    @onnx9000.jit
    def linear(x, w):
        """linear docstring."""

        return x @ w

    x = onnx9000.Tensor(shape=(32, 128), dtype=DType.FLOAT32, name="x")
    w = onnx9000.Parameter(shape=(128, 64), dtype=DType.FLOAT32, name="w")

    builder = linear(x, w)

    out_path = temp_dir / "linear.onnx"
    onnx9000.to_onnx(builder, out_path)

    assert out_path.exists()

    model_proto = onnx_pb2.ModelProto()
    with open(out_path, "rb") as f:
        model_proto.ParseFromString(f.read())

    assert len(model_proto.graph.node) == 1
    assert model_proto.graph.node[0].op_type == "MatMul"
    # Parameter should be in initializers
    assert len(model_proto.graph.initializer) == 1
    assert model_proto.graph.initializer[0].name == "w"


def test_export_conv_with_attrs(temp_dir: Path):
    """test_export_conv_with_attrs docstring."""

    @onnx9000.jit
    def conv_model(x, w):
        """conv_model docstring."""

        return onnx9000.ops.conv(x, w, strides=[2, 2], pads=[1, 1, 1, 1])

    x = onnx9000.Tensor(shape=(1, 3, 224, 224), dtype=DType.FLOAT32, name="x")
    w = onnx9000.Parameter(shape=(64, 3, 7, 7), dtype=DType.FLOAT32, name="w")

    builder = conv_model(x, w)
    out_path = temp_dir / "conv.onnx"
    onnx9000.to_onnx(builder, out_path)

    model_proto = onnx_pb2.ModelProto()
    with open(out_path, "rb") as f:
        model_proto.ParseFromString(f.read())

    assert len(model_proto.graph.node) == 1
    node = model_proto.graph.node[0]
    assert node.op_type == "Conv"

    attrs = {attr.name: attr for attr in node.attribute}
    assert "strides" in attrs
    assert "pads" in attrs
    assert list(attrs["strides"].ints) == [2, 2]
    assert list(attrs["pads"].ints) == [1, 1, 1, 1]
