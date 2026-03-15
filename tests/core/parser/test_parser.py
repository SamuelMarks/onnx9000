"""Module providing core logic and structural definitions."""

from pathlib import Path
import onnx9000
from onnx9000.core.dtypes import DType


def test_parse_simple_model(temp_dir: Path):
    """Tests the test parse simple model functionality."""

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
    ir_graph = onnx9000.core.parser.core.load(out_path)
    assert ir_graph.name == "simple_model"
    assert "x" in ir_graph.inputs
    assert "w" in ir_graph.initializers
    assert "w" in ir_graph.tensors
    assert ir_graph.tensors["w"].shape == (20, 30)
    assert ir_graph.tensors["w"].is_initializer is True
    assert "x" in ir_graph.tensors
    assert ir_graph.tensors["x"].shape == (10, 20)
    assert len(ir_graph.nodes) == 2
    assert ir_graph.nodes[0].op_type == "MatMul"
    assert ir_graph.nodes[1].op_type == "Relu"


def test_fusions(temp_dir: Path):
    """Tests the test fusions functionality."""

    @onnx9000.jit
    def fusion_model(x, w, b):
        """Provides fusion model functionality and verification."""
        h = x @ w
        return h + b

    x = onnx9000.Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")
    w = onnx9000.Parameter(shape=(20, 30), dtype=DType.FLOAT32, name="w")
    b = onnx9000.Parameter(shape=(30,), dtype=DType.FLOAT32, name="b")
    builder = fusion_model(x, w, b)
    out_path = temp_dir / "fusion.onnx"
    onnx9000.to_onnx(builder, out_path)
    ir_graph = onnx9000.core.parser.core.load(out_path)
    onnx9000.core.parser.passes.optimize(ir_graph)
    assert len(ir_graph.nodes) == 1
    assert ir_graph.nodes[0].op_type == "Gemm"
    assert len(ir_graph.nodes[0].inputs) == 3
    assert "x" in ir_graph.nodes[0].inputs
    assert "w" in ir_graph.nodes[0].inputs
    assert "b" in ir_graph.nodes[0].inputs


def test_fuse_transpose(temp_dir: Path):
    """Tests the test fuse transpose functionality."""

    @onnx9000.jit
    def trans_model(x):
        """Provides trans model functionality and verification."""
        h = onnx9000.core.ops.transpose(x)
        return onnx9000.core.ops.transpose(h)

    x = onnx9000.Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")
    builder = trans_model(x)
    out_path = temp_dir / "trans.onnx"
    onnx9000.to_onnx(builder, out_path)
    ir_graph = onnx9000.core.parser.core.load(out_path)
    onnx9000.core.parser.passes.optimize(ir_graph)
    assert len(ir_graph.nodes) == 0


def test_plan_memory(temp_dir: Path):
    """Tests the test plan memory functionality."""

    @onnx9000.jit
    def multi_step(a, b):
        """Provides multi step functionality and verification."""
        c = a + b
        d = c * a
        e = onnx9000.core.ops.relu(d)
        return e

    a = onnx9000.Tensor(shape=(5,), dtype=DType.FLOAT32, name="a")
    b = onnx9000.Tensor(shape=(5,), dtype=DType.FLOAT32, name="b")
    builder = multi_step(a, b)
    out_path = temp_dir / "multi_step.onnx"
    onnx9000.to_onnx(builder, out_path)
    ir_graph = onnx9000.core.parser.core.load(out_path)
    onnx9000.core.parser.plan_memory(ir_graph)
    c_name = builder.nodes[0].outputs[0].name
    d_name = builder.nodes[1].outputs[0].name
    e_name = builder.nodes[2].outputs[0].name
    c_buf = ir_graph.tensors[c_name].buffer_id
    d_buf = ir_graph.tensors[d_name].buffer_id
    e_buf = ir_graph.tensors[e_name].buffer_id
    buffers = {c_buf, d_buf, e_buf}
    assert len(buffers) < 3
    for _name, tensor in ir_graph.tensors.items():
        if not tensor.is_initializer and _name not in ir_graph.inputs:
            assert tensor.buffer_id is not None
            assert tensor.lifespan[0] <= tensor.lifespan[1]
            assert tensor.lifespan[1] > -1
