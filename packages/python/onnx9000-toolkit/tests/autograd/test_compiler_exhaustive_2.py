"""Tests the compiler exhaustive 2 module functionality."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import (
    AOTBuilder,
    AutogradEngine,
    apply_automatic_mixed_precision,
    build_backward_graph,
    estimate_batch_size_limit,
    inject_custom_loss_subgraph,
    inject_memcpy_boundaries,
    scale_backward_graph_for_mixed_precision,
    set_eval_mode,
)


def test_add_loss_subgraph_missing_tensor():
    """Tests the add loss subgraph missing tensor functionality."""
    g = Graph("g")
    loss_g = Graph("loss_g")
    loss_g.add_node(Node("Identity", ["t_loss"], ["t_loss_out"], name="id"))
    loss_g.add_tensor(Tensor("t_loss", [1], "float32", requires_grad=True))
    inject_custom_loss_subgraph(g, loss_g, {"t_loss": "t_mapped"})

    assert "t_mapped" in g.tensors


def test_amp_skip_non_float32():
    """Tests the amp skip non float32 functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("init_non_f32", [1], "int64"))
    g.initializers.append("init_non_f32")
    g.add_tensor(Tensor("inp_non_f32", [1], "int64"))
    g.inputs.append("inp_non_f32")
    apply_automatic_mixed_precision(g, "float16")
    assert "init_non_f32_cast_float16" not in g.tensors


def test_inject_memcpy_boundaries_skip_non_float32():
    """Tests the inject memcpy boundaries skip non float32 functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("init_non_f32", [1], "int64"))
    g.initializers.append("init_non_f32")
    g.add_tensor(Tensor("inp_non_f32", [1], "int64"))
    g.inputs.append("inp_non_f32")
    inject_memcpy_boundaries(g)
    assert "init_non_f32_cast_float16" not in g.tensors


def test_scale_loss_gradient_existing():
    """Tests the scale loss gradient existing functionality."""
    g = Graph("g")
    g.inputs.append("grad_loss")
    scale_backward_graph_for_mixed_precision(g, 2.0)
    assert "grad_loss" not in g.inputs
    assert any(n.name == "grad_loss_scaled_c" for n in g.nodes)


def test_set_eval_mode_normal_node():
    """Tests the set eval mode normal node functionality."""
    g = Graph("g")
    g.add_node(Node("Relu", ["x"], ["y"], name="relu1"))
    eval_g = set_eval_mode(g)
    assert any(n.op_type == "Relu" for n in eval_g.nodes)


def test_autograd_engine_retained_grads():
    """Tests the autograd engine retained grads functionality."""
    engine = AutogradEngine()
    engine.retain_grad("x")
    engine.retain_grad("x")  # should not add twice
    assert engine._retained_grads == ["x"]


def test_build_backward_graph_retained_grads():
    """Tests the build backward graph retained grads functionality."""
    fwd = Graph("fwd")
    fwd.add_node(Node("Relu", ["x"], ["y"], name="relu1"))
    fwd.add_tensor(Tensor("x", [1], "float32", requires_grad=True))
    fwd.add_tensor(Tensor("y", [1], "float32", requires_grad=True))
    fwd.outputs.append("y")
    bwd = build_backward_graph(fwd, retained_grads=["x"])
    assert any(out.startswith("grad_x") for out in bwd.outputs)


def test_build_backward_graph_no_vjp():
    """Tests the build backward graph no vjp functionality."""
    fwd = Graph("fwd")
    fwd.add_node(Node("UnknownOp", ["x"], ["y"], name="unk"))
    fwd.add_tensor(Tensor("y", [1], "float32", requires_grad=True))
    fwd.outputs.append("y")
    with pytest.raises(RuntimeError):
        build_backward_graph(fwd)


def test_estimate_batch_size_limit_positive():
    """Tests the estimate batch size limit positive functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("large_in", (1024, 1024, 1024), "float32", is_initializer=True))
    g.initializers.append("large_in")
    g.add_node(Node("Add", ["large_in", "large_in"], ["large_out"], name="add"))
    res = estimate_batch_size_limit(g, 8000)
    assert res >= 1


def test_aot_builder_init_no_opset():
    """Tests the aot builder init no opset functionality."""
    g = Graph("g")
    if hasattr(g, "opset_imports"):
        delattr(g, "opset_imports")
    builder = AOTBuilder(g)
    assert builder.fwd_graph.opset_imports["ai.onnx"] == 15


def test_aot_builder_accumulate_gradient():
    """Tests the aot builder accumulate gradient functionality."""
    g = Graph("g")
    g.add_node(Node("Relu", ["x"], ["y"], name="relu"))
    g.outputs.append("y")
    g.initializers.append("x")
    g.tensors["x"] = Tensor("x", [1], "float32", requires_grad=True)
    g.tensors["y"] = Tensor("y", [1], "float32", requires_grad=True)
    builder = AOTBuilder(g)
    bwd_only = builder.build_accumulate_gradient_graph()
    assert any(out.startswith("grad_") for out in bwd_only.outputs)
