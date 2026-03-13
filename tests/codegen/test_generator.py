"""Module docstring."""

from pathlib import Path

import onnx9000
from onnx9000.codegen import Generator
from onnx9000.dtypes import DType


def test_generate_simple_model(temp_dir: Path):
    """test_generate_simple_model docstring."""

    @onnx9000.jit
    def simple_model(x, w):
        """simple_model docstring."""

        h = x @ w
        return onnx9000.ops.relu(h)

    x = onnx9000.Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")
    w = onnx9000.Parameter(shape=(20, 30), dtype=DType.FLOAT32, name="w")

    builder = simple_model(x, w)

    out_path = temp_dir / "simple.onnx"
    onnx9000.to_onnx(builder, out_path)

    ir_graph = onnx9000.parser.load(out_path)
    onnx9000.parser.plan_memory(ir_graph)

    generator = Generator(ir_graph, class_name="MyTestModel")
    cpp_code = generator.generate()

    print(cpp_code)

    assert "class MyTestModel {" in cpp_code
    assert "std::expected<void, std::string> forward(float* x_ptr, float* " in cpp_code
    assert "_out_ptr_0) noexcept {" in cpp_code
    assert "// MatMul" in cpp_code
    assert "// Unary Op: Relu" in cpp_code
    assert "_arena.resize(" in cpp_code
