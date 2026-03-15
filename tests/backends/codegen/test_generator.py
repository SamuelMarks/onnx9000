"""Module providing core logic and structural definitions."""

from pathlib import Path
from unittest import mock
import onnx9000
from onnx9000.backends.codegen import Generator
from onnx9000.core.dtypes import DType
from onnx9000.core import config


def test_generate_simple_model(temp_dir: Path):
    """Tests the test generate simple model functionality."""

    @onnx9000.jit
    def simple_model(x, w):
        """Provides simple model functionality and verification."""
        h = x @ w
        return onnx9000.core.ops.relu(h)

    x = onnx9000.Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")
    w = onnx9000.Parameter(shape=(20, 30), dtype=DType.FLOAT32, name="w")
    builder = simple_model(x, w)
    out_path = temp_dir / "simple.onnx"
    onnx9000.to_onnx(builder, out_path)
    ir_graph = onnx9000.core.parser.load(out_path)
    onnx9000.core.parser.plan_memory(ir_graph)
    generator = Generator(ir_graph, class_name="MyTestModel")
    cpp_code = generator.generate()
    assert "class MyTestModel {" in cpp_code
    assert "std::expected<void, std::string> forward(float* x_ptr, float* " in cpp_code
    assert "_out_ptr_0) noexcept {" in cpp_code
    assert "// MatMul" in cpp_code
    assert "// Unary Op: Relu" in cpp_code
    assert "_arena.resize(" in cpp_code


def test_generator_cuda_and_no_buffer_id(temp_dir: Path):
    """Test generating with CUDA and no buffer id."""

    @onnx9000.jit
    def simple_model(x):
        """Provides simple model functionality and verification."""
        return onnx9000.core.ops.relu(x)

    x = onnx9000.Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")
    builder = simple_model(x)
    out_path = temp_dir / "simple_cuda.onnx"
    onnx9000.to_onnx(builder, out_path)
    ir_graph = onnx9000.core.parser.load(out_path)
    with mock.patch.object(config, "ONNX9000_USE_CUDA", True):
        generator = Generator(ir_graph, class_name="CudaModel")
        cpp_code = generator.generate()
    assert "CudaModel" in cpp_code
    assert "onnx9000::CudaBuffer" in cpp_code


def test_generator_multi_output(temp_dir):
    """Provides semantic functionality and verification."""
    from onnx9000.core.ir import Graph
    from onnx9000.core.dtypes import DType
    import onnx9000

    ir_graph = Graph(name="multi")
    ir_graph.outputs = ["o1", "o2"]
    ir_graph.tensors["o1"] = onnx9000.Tensor(shape=(1,), dtype=DType.FLOAT32, name="o1")
    ir_graph.tensors["o2"] = onnx9000.Tensor(shape=(1,), dtype=DType.FLOAT32, name="o2")
    ir_graph.tensors["o1"].buffer_id = 1
    ir_graph.tensors["o2"].buffer_id = 2
    generator = Generator(ir_graph, class_name="Multi")
    cpp_code = generator.generate()


def test_generator_zero_output(temp_dir):
    """Provides semantic functionality and verification."""
    from onnx9000.core.ir import Graph

    ir_graph = Graph(name="empty")
    generator = Generator(ir_graph, class_name="Empty")
    cpp_code = generator.generate()
