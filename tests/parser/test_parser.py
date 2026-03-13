"""Module docstring."""

from pathlib import Path

import onnx9000
from onnx9000.dtypes import DType


def test_parse_simple_model(temp_dir: Path):
    """test_parse_simple_model docstring."""

    # First export a simple model
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

    # Now parse it back
    ir_graph = onnx9000.parser.core.load(out_path)

    assert ir_graph.name == "simple_model"
    assert "x" in ir_graph.inputs
    assert "w" in ir_graph.initializers

    # Check Tensors
    assert "w" in ir_graph.tensors
    assert ir_graph.tensors["w"].shape == (20, 30)
    assert ir_graph.tensors["w"].is_initializer is True

    assert "x" in ir_graph.tensors
    assert ir_graph.tensors["x"].shape == (10, 20)

    # Check Nodes
    assert len(ir_graph.nodes) == 2
    assert ir_graph.nodes[0].op_type == "MatMul"
    assert ir_graph.nodes[1].op_type == "Relu"


def test_fusions(temp_dir: Path):
    """test_fusions docstring."""

    @onnx9000.jit
    def fusion_model(x, w, b):
        """fusion_model docstring."""
        h = x @ w
        return h + b

    x = onnx9000.Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")
    w = onnx9000.Parameter(shape=(20, 30), dtype=DType.FLOAT32, name="w")
    b = onnx9000.Parameter(shape=(30,), dtype=DType.FLOAT32, name="b")

    builder = fusion_model(x, w, b)
    out_path = temp_dir / "fusion.onnx"
    onnx9000.to_onnx(builder, out_path)

    ir_graph = onnx9000.parser.core.load(out_path)

    # Run optimizer (fusion)
    onnx9000.parser.passes.optimize(ir_graph)

    # We expect MatMul + Add to become Gemm
    assert len(ir_graph.nodes) == 1
    assert ir_graph.nodes[0].op_type == "Gemm"
    assert len(ir_graph.nodes[0].inputs) == 3
    assert "x" in ir_graph.nodes[0].inputs
    assert "w" in ir_graph.nodes[0].inputs
    assert "b" in ir_graph.nodes[0].inputs


def test_fuse_transpose(temp_dir: Path):
    """test_fuse_transpose docstring."""

    @onnx9000.jit
    def trans_model(x):
        """trans_model docstring."""
        h = onnx9000.ops.transpose(x)
        return onnx9000.ops.transpose(h)

    x = onnx9000.Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")
    builder = trans_model(x)
    out_path = temp_dir / "trans.onnx"
    onnx9000.to_onnx(builder, out_path)

    ir_graph = onnx9000.parser.core.load(out_path)
    onnx9000.parser.passes.optimize(ir_graph)

    # They should cancel out. 0 nodes remain.
    assert len(ir_graph.nodes) == 0


def test_plan_memory(temp_dir: Path):
    """test_plan_memory docstring."""

    @onnx9000.jit
    def multi_step(a, b):
        """multi_step docstring."""

        c = a + b
        d = c * a
        e = onnx9000.ops.relu(d)
        return e

    a = onnx9000.Tensor(shape=(5,), dtype=DType.FLOAT32, name="a")
    b = onnx9000.Tensor(shape=(5,), dtype=DType.FLOAT32, name="b")

    builder = multi_step(a, b)
    out_path = temp_dir / "multi_step.onnx"
    onnx9000.to_onnx(builder, out_path)

    ir_graph = onnx9000.parser.core.load(out_path)

    # Run memory planner
    onnx9000.parser.plan_memory(ir_graph)

    # Check lifespans
    # a and b are inputs, created at idx 0
    # node 1: a + b -> c (c created at 1)
    # node 2: c * a -> d (d created at 2, c dies at 2, a dies at 2)
    # node 3: relu(d) -> e (e created at 3, d dies at 3)
    # outputs: e (dies at end_idx = 4)

    # c lifespan is [1, 2]
    # d lifespan is [2, 3]
    # e lifespan is [3, 4]

    # We rely on the naming generation logic
    c_name = builder.nodes[0].outputs[0].name
    d_name = builder.nodes[1].outputs[0].name
    e_name = builder.nodes[2].outputs[0].name

    c_buf = ir_graph.tensors[c_name].buffer_id
    d_buf = ir_graph.tensors[d_name].buffer_id
    e_buf = ir_graph.tensors[e_name].buffer_id

    # Check that there is at least some reuse happening!
    # Max active overlapping tensors is 2 at any time, so buffers should be max 2
    buffers = {c_buf, d_buf, e_buf}
    assert (
        len(buffers) < 3
    )  # Proves that reuse happened (either c == e or d == e or c == d depending on traversal)

    for _name, tensor in ir_graph.tensors.items():
        if not tensor.is_initializer and _name not in ir_graph.inputs:
            assert tensor.buffer_id is not None
            assert tensor.lifespan[0] <= tensor.lifespan[1]
            assert tensor.lifespan[1] > -1
