"""Tests the compiler exhaustive module functionality."""

import json
import os

import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import (
    AOTBuilder,
    AutogradEngine,
    analytical_jacobian,
    apply_automatic_mixed_precision,
    apply_peft_config,
    build_backward_graph,
    cast_gradients_to_fp32,
    enforce_webgpu_limits,
    ensure_no_microsoft_opsets,
    estimate_batch_size_limit,
    freeze_layers,
    hessian_vector_product,
    implement_activation_checkpointing,
    inject_bitfit,
    inject_explicit_yield_nodes,
    inject_inplace_hints,
    inject_memcpy_boundaries,
    inject_nan_inf_bypass,
    load_training_checkpoint,
    optimize_intermediate_casts,
    optimize_memory_reuse,
    profile_lora_memory_savings,
    save_lora_adapters,
    save_training_checkpoint,
    set_eval_mode,
    setup_incremental_stream,
    track_vram_usage,
    validate_amp_rules,
    verify_no_circular_references,
)


def test_save_training_checkpoint(tmp_path):
    """Tests the save training checkpoint functionality."""
    graph = Graph(name="g")
    graph.initializers.append("w1")
    graph.add_tensor(Tensor(name="w1", shape=(10,), dtype="float32", requires_grad=True))

    filepath = tmp_path / "ckpt.json"
    save_training_checkpoint(graph, str(filepath))

    with open(filepath) as f:
        data = json.load(f)
    assert data == {"w1": "tensor_data_placeholder"}


def test_save_lora_adapters(tmp_path):
    """Tests the save lora adapters functionality."""
    graph = Graph(name="g")
    graph.initializers.extend(["lora_a_1", "lora_b_1", "w1"])

    filepath = tmp_path / "lora.json"
    save_lora_adapters(graph, str(filepath))

    with open(filepath) as f:
        data = json.load(f)
    assert set(data["adapters"]) == {"lora_a_1", "lora_b_1"}


def test_inject_memcpy_boundaries():
    """Tests the inject memcpy boundaries functionality."""
    graph = Graph(name="g")
    inject_memcpy_boundaries(graph)


def test_validate_amp_rules():
    """Tests the validate amp rules functionality."""
    graph = Graph(name="g")
    validate_amp_rules(graph)


def test_apply_automatic_mixed_precision():
    """Tests the apply automatic mixed precision functionality."""
    graph = Graph(name="g")
    graph.initializers.append("w1")
    graph.add_tensor(Tensor(name="w1", shape=(10,), dtype="float32"))
    graph.inputs.append("w1")
    graph.inputs.append("x")
    graph.add_tensor(Tensor(name="x", shape=(10,), dtype="float32"))
    graph.add_node(Node("MatMul", ["x", "w1"], ["y"], name="m1"))

    apply_automatic_mixed_precision(graph, "float16")

    assert "w1_cast_float16" in graph.tensors
    assert "x_cast_float16" in graph.tensors

    # Also test bfloat16
    graph2 = Graph(name="g2")
    graph2.initializers.append("w1")
    graph2.add_tensor(Tensor(name="w1", shape=(10,), dtype="float32"))
    graph2.inputs.append("w1")
    graph2.inputs.append("x")
    graph2.add_tensor(Tensor(name="x", shape=(10,), dtype="float32"))
    graph2.add_node(Node("MatMul", ["x", "w1"], ["y"], name="m1"))

    apply_automatic_mixed_precision(graph2, "bfloat16")
    assert "w1_cast_bfloat16" in graph2.tensors


def test_cast_gradients_to_fp32():
    """Tests the cast gradients to fp32 functionality."""
    graph = Graph(name="g")
    graph.outputs.append("grad_w1")
    graph.add_tensor(Tensor(name="grad_w1", shape=(10,), dtype="float16"))
    graph.add_node(Node("Add", ["x", "y"], ["grad_w1"], name="n1"))

    cast_gradients_to_fp32(graph)
    assert "grad_w1_fp32" in graph.tensors
    assert any(n.op_type == "Cast" and n.outputs == ["grad_w1"] for n in graph.nodes)


def test_optimize_intermediate_casts():
    """Tests the optimize intermediate casts functionality."""
    graph = Graph(name="g")
    graph.add_tensor(Tensor(name="a", shape=(1,), dtype="float32"))
    graph.add_tensor(Tensor(name="b", shape=(1,), dtype="float16"))
    graph.add_tensor(Tensor(name="c", shape=(1,), dtype="float32"))

    # Cast -> Cast
    graph.add_node(Node("Cast", ["a"], ["b"], name="c1"))
    graph.add_node(Node("Cast", ["b"], ["c"], name="c2"))

    optimize_intermediate_casts(graph)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].inputs == ["a"]


def test_implement_activation_checkpointing():
    """Tests the implement activation checkpointing functionality."""
    implement_activation_checkpointing(Graph("g"))


def test_setup_incremental_stream():
    """Tests the setup incremental stream functionality."""
    setup_incremental_stream(Graph("g"), "ws://localhost")


def test_load_training_checkpoint(tmp_path):
    """Tests the load training checkpoint functionality."""
    graph = Graph("g")
    graph.add_tensor(Tensor("w1", (1,), "float32"))
    filepath = tmp_path / "ckpt.json"
    with open(filepath, "w") as f:
        json.load = lambda f: {"w1": "data"}  # mock
        json.dump({"w1": "data"}, f)

    load_training_checkpoint(graph, str(filepath))


def test_set_eval_mode():
    """Tests the set eval mode functionality."""
    graph = Graph("g")
    graph.inputs.append("x")
    graph.outputs.append("y")
    graph.initializers.append("w")
    graph.add_tensor(Tensor("x", (1,), "float32"))

    graph.add_node(Node("Dropout", ["a"], ["b", "mask"], name="d1"))
    graph.add_node(Node("BatchNormalization", ["a"], ["b"], {"training_mode": 1}, name="bn1"))

    eval_g = set_eval_mode(graph)
    assert any(n.op_type == "Identity" for n in eval_g.nodes)
    assert any(n.op_type == "ConstantOfShape" for n in eval_g.nodes)
    assert any(
        n.op_type == "BatchNormalization" and n.attributes.get("training_mode") == 0
        for n in eval_g.nodes
    )


def test_freeze_layers():
    """Tests the freeze layers functionality."""
    graph = Graph("g")
    graph.add_tensor(Tensor("w1", (1,), "float32", requires_grad=True))
    freeze_layers(graph, ["w1"])
    assert not graph.tensors["w1"].requires_grad


def test_inject_bitfit():
    """Tests the inject bitfit functionality."""
    graph = Graph("g")
    graph.initializers.extend(["w1", "bias1", "w2"])
    graph.add_tensor(Tensor("w1", (10, 10), "float32", requires_grad=True))
    graph.add_tensor(Tensor("bias1", (10,), "float32", requires_grad=True))
    graph.add_tensor(Tensor("w2", (10, 10), "float32", requires_grad=True))

    inject_bitfit(graph)
    assert not graph.tensors["w1"].requires_grad
    assert graph.tensors["bias1"].requires_grad
    assert not graph.tensors["w2"].requires_grad


def test_apply_peft_config():
    """Tests the apply peft config functionality."""
    graph = Graph("g")
    graph.initializers.extend(["w1", "bias1", "prompt_w"])
    graph.add_tensor(Tensor("w1", (10, 10), "float32", requires_grad=True))
    graph.add_tensor(Tensor("bias1", (10,), "float32", requires_grad=True))
    graph.add_tensor(Tensor("prompt_w", (10,), "float32", requires_grad=True))

    apply_peft_config(graph, {"peft_type": "bitfit"})
    assert not graph.tensors["w1"].requires_grad
    assert graph.tensors["bias1"].requires_grad

    graph2 = Graph("g2")
    graph2.initializers.extend(["w1", "prompt_w"])
    graph2.add_tensor(Tensor("w1", (10, 10), "float32", requires_grad=True))
    graph2.add_tensor(Tensor("prompt_w", (10,), "float32", requires_grad=True))

    apply_peft_config(graph2, {"peft_type": "prompt_tuning"})
    assert not graph2.tensors["w1"].requires_grad
    assert graph2.tensors["prompt_w"].requires_grad

    apply_peft_config(graph2, {"peft_type": "lora", "target_modules": ["w1"]})


def test_autograd_engine_no_grad():
    """Tests the autograd engine no grad functionality."""
    engine = AutogradEngine()
    with engine.no_grad():
        assert engine._no_grad


def test_inject_explicit_yield_nodes():
    """Tests the inject explicit yield nodes functionality."""
    inject_explicit_yield_nodes(Graph("g"))


def test_verify_no_circular_references():
    """Tests the verify no circular references functionality."""
    g = Graph("g")
    g.toposort = lambda: None
    verify_no_circular_references(g)


def test_inject_inplace_hints():
    """Tests the inject inplace hints functionality."""
    inject_inplace_hints(Graph("g"))


def test_optimize_memory_reuse():
    """Tests the optimize memory reuse functionality."""
    optimize_memory_reuse(Graph("g"))


def test_inject_nan_inf_bypass():
    """Tests the inject nan inf bypass functionality."""
    graph = Graph("g")
    inject_nan_inf_bypass(graph, [])
    inject_nan_inf_bypass(graph, ["grad_1"])
    assert "grad_1_invalid" in graph.outputs or any(
        n.outputs == ["grad_1_invalid"] for n in graph.nodes
    )

    graph2 = Graph("g2")
    inject_nan_inf_bypass(graph2, ["grad_1", "grad_2"])
    assert any(n.op_type == "ReduceOr" for n in graph2.nodes)


def test_build_backward_graph_custom_domain():
    """Tests the build backward graph custom domain functionality."""
    graph = Graph("g")
    graph.inputs.append("x")
    graph.outputs.append("y")
    graph.add_tensor(Tensor("x", (1,), "float32", requires_grad=True))
    graph.add_tensor(Tensor("y", (1,), "float32", requires_grad=True))

    node = Node("CustomOp", ["x"], ["y"], name="n1")
    node.domain = "my.domain"
    graph.add_node(node)

    with pytest.raises(RuntimeError, match="belongs to custom domain"):
        build_backward_graph(graph)


def test_hessian_vector_product():
    """Tests the hessian vector product functionality."""
    graph = Graph("g")
    graph.inputs.append("x")
    graph.outputs.append("y")
    graph.add_tensor(Tensor("x", (1,), "float32", requires_grad=True))
    graph.add_tensor(Tensor("y", (1,), "float32", requires_grad=True))
    graph.add_node(Node("Relu", ["x"], ["y"], name="n1"))

    hvp = hessian_vector_product(graph, ["v"])
    assert hvp is not None


def test_analytical_jacobian():
    """Tests the analytical jacobian functionality."""
    graph = Graph("g")
    graph.inputs.append("x")
    graph.outputs.append("y")
    graph.add_tensor(Tensor("x", (1,), "float32", requires_grad=True))
    graph.add_tensor(Tensor("y", (1,), "float32", requires_grad=True))
    graph.add_node(Node("Relu", ["x"], ["y"], name="n1"))

    jac = analytical_jacobian(graph)
    assert jac is not None


def test_ensure_no_microsoft_opsets():
    """Tests the ensure no microsoft opsets functionality."""
    graph = Graph("g")
    node = Node("Op", [], [], name="n1")
    node.domain = "com.microsoft"
    graph.add_node(node)

    with pytest.raises(ValueError):
        ensure_no_microsoft_opsets(graph)

    graph2 = Graph("g2")
    graph2.opset_imports = {"com.microsoft": 1}
    with pytest.raises(ValueError):
        ensure_no_microsoft_opsets(graph2)


def test_enforce_webgpu_limits():
    """Tests the enforce webgpu limits functionality."""
    graph = Graph("g")
    graph.add_tensor(Tensor("t1", (10000, 10000, 10000), "float32"))
    with pytest.raises(ValueError):
        enforce_webgpu_limits(graph)


def test_track_vram_usage():
    """Tests the track vram usage functionality."""
    graph = Graph("g")
    graph.add_tensor(Tensor("t1", (10, 10), "float32"))
    vram = track_vram_usage(graph)
    assert vram >= 0


def test_profile_lora_memory_savings():
    """Tests the profile lora memory savings functionality."""
    graph = Graph("g")
    # full params = 0
    assert profile_lora_memory_savings(graph) == 0.0

    graph.add_tensor(Tensor("lora_a", (10, 10), "float32", requires_grad=True))
    graph.add_tensor(Tensor("w1", (100, 100), "float32", requires_grad=True))
    savings = profile_lora_memory_savings(graph)
    assert savings > 0.0


def test_estimate_batch_size_limit():
    """Tests the estimate batch size limit functionality."""
    graph = Graph("g")
    graph.add_tensor(Tensor("t1", (10, 10), "float32"))
    limit = estimate_batch_size_limit(graph, 4000)
    assert limit >= 1

    # zero vram case
    graph2 = Graph("g2")
    limit2 = estimate_batch_size_limit(graph2, 4000)
    assert limit2 == 1


def test_aot_builder():
    """Tests the aot builder functionality."""
    graph = Graph("g")
    graph.inputs.append("x")
    graph.outputs.append("y")
    graph.initializers.append("w1")
    graph.initializers.append("lora_a")
    graph.add_tensor(Tensor("x", (10,), "float32"))
    graph.add_tensor(Tensor("y", (10,), "float32", requires_grad=True))
    graph.add_tensor(Tensor("w1", (10,), "float32", requires_grad=True))
    graph.add_tensor(Tensor("lora_a", (10,), "float32", requires_grad=True))

    graph.add_node(Node("Relu", ["x"], ["y"], name="relu"))

    builder = AOTBuilder(graph)

    # build_accumulate_gradient_graph
    accum_graph = builder.build_accumulate_gradient_graph()
    assert accum_graph is not None

    # fake optimizer gen
    def fake_opt(g, lr, params):
        """Tests the fake opt functionality."""
        for p in params:
            g.outputs.append(f"{p}_new")

    # build_lora_optimizer_graph
    lora_opt = builder.build_lora_optimizer_graph(fake_opt, "lr")
    assert "lora_a_new" in lora_opt.outputs

    # build_optimizer_graph
    opt = builder.build_optimizer_graph(fake_opt, "lr")
    assert "w1_new" in opt.outputs

    # build_gradient_deltas_graph
    deltas = builder.build_gradient_deltas_graph(fake_opt, "lr")
    assert "delta_w1" in deltas.outputs
