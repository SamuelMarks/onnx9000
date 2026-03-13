"""Module docstring."""

import onnx9000
from onnx9000.dtypes import DType


def test_tensor_creation():
    """test_tensor_creation docstring."""

    t = onnx9000.Tensor(shape=(1, 3, 224, 224), dtype=DType.FLOAT32, name="input_1")
    assert t.shape == (1, 3, 224, 224)
    assert t.dtype == DType.FLOAT32
    assert t.name == "input_1"


def test_tracing_context():
    """test_tracing_context docstring."""

    t1 = onnx9000.Tensor(shape=(10,), dtype=DType.FLOAT32)
    t2 = onnx9000.Tensor(shape=(10,), dtype=DType.FLOAT32)

    builder = onnx9000.GraphBuilder(name="test_graph")

    with onnx9000.Tracing(builder):
        t3 = t1 + t2
        t4 = t3 * t1
        t5 = onnx9000.ops.relu(t4)

    assert len(builder.nodes) == 3
    assert builder.nodes[0].op_type == "Add"
    assert builder.nodes[1].op_type == "Mul"
    assert builder.nodes[2].op_type == "Relu"

    assert t3.shape == (10,)
    assert t5.shape == (10,)


def test_jit_decorator():
    """test_jit_decorator docstring."""

    @onnx9000.jit
    def simple_model(x, w):
        """simple_model docstring."""
        # Linear layer
        h = x @ w
        return onnx9000.ops.relu(h)

    x = onnx9000.Tensor(shape=(32, 128), dtype=DType.FLOAT32, name="x")
    w = onnx9000.Parameter(shape=(128, 64), dtype=DType.FLOAT32, name="w")

    builder = simple_model(x, w)

    assert isinstance(builder, onnx9000.GraphBuilder)
    assert builder.name == "simple_model"
    assert len(builder.nodes) == 2
    assert builder.nodes[0].op_type == "MatMul"
    assert builder.nodes[1].op_type == "Relu"

    # Check outputs and inferred shapes
    assert len(builder.outputs) == 1
    assert builder.outputs[0].shape == (32, 64)


def test_tracing_dtypes():
    """test_tracing_dtypes docstring."""
    import numpy as np

    t1 = onnx9000.Tensor(shape=(10,), dtype=DType.FLOAT32)
    builder = onnx9000.GraphBuilder(name="test_dtypes_graph")
    with onnx9000.Tracing(builder):
        onnx9000.ops.add(t1, np.array([1], dtype=np.int32))
        onnx9000.ops.add(t1, np.array([1.0], dtype=np.float64))
        onnx9000.ops.add(t1, np.array([True], dtype=bool))
