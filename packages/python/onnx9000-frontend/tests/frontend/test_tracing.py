import onnx9000
import onnx9000.core.ops

"Module providing core logic and structural definitions."
from onnx9000.core.dtypes import DType
from onnx9000.frontend.frontend.tensor import Tensor
from onnx9000.frontend.frontend.tracer import script as jit


def test_tensor_creation() -> None:
    """Tests the test tensor creation functionality."""
    t = Tensor(shape=(1, 3, 224, 224), dtype=DType.FLOAT32, name="input_1")
    assert t.shape == (1, 3, 224, 224)
    assert t.dtype == DType.FLOAT32
    assert t.name == "input_1"


def test_tracing_context() -> None:
    """Tests the test tracing context functionality."""
    t1 = Tensor(shape=(10,), dtype=DType.FLOAT32)
    t2 = Tensor(shape=(10,), dtype=DType.FLOAT32)
    builder = onnx9000.frontend.frontend.builder.GraphBuilder(name="test_graph")
    with onnx9000.frontend.frontend.builder.Tracing(builder):
        t3 = t1 + t2
        t4 = t3 * t1
        t5 = t4.relu()
    assert len(builder.nodes) == 3
    assert builder.nodes[2].op_type == "Relu"
    assert t3.shape == (10,)
    assert t5.shape == (10,)


def test_jit_decorator() -> None:
    """Tests the test jit decorator functionality."""

    def simple_model(x, w):
        """Tests the simple_model functionality."""
        h = x @ w
        return h.relu()

    x = Tensor(shape=(32, 128), dtype=DType.FLOAT32, name="x")
    w = onnx9000.frontend.frontend.tensor.Parameter(shape=(128, 64), dtype=DType.FLOAT32, name="w")
    builder = jit(simple_model, x, w)
    assert isinstance(builder, onnx9000.frontend.frontend.builder.GraphBuilder)
    assert builder.name == "simple_model"


def test_tracing_dtypes() -> None:
    """Tests the test tracing dtypes functionality."""
    import numpy as np

    t1 = Tensor(shape=(10,), dtype=DType.FLOAT32)
    builder = onnx9000.frontend.frontend.builder.GraphBuilder(name="test_dtypes_graph")
    with onnx9000.frontend.frontend.builder.Tracing(builder):
        t1 + np.array([1], dtype=np.int32)
        t1 + np.array([1.0], dtype=np.float64)
        t1 + np.array([True], dtype=bool)
