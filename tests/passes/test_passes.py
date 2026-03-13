"""Module docstring."""

from onnx9000.ir import Graph, Node
from onnx9000.passes import (
    dead_code_elimination,
    constant_folding,
    fuse_linear_activation,
    fuse_consecutive_transpose,
    fuse_matmul_add,
    enforce_opset_18,
    apply_opset_fallbacks,
    detect_cycles,
    flatten_subgraphs,
    optimize_broadcasting,
    transform_nchw_to_nhwc,
    transform_nhwc_to_nchw,
    inject_probes,
    optimize_for_webgpu,
    polyfill_webgpu_unsupported,
    insert_qat_nodes,
    convert_to_int8,
    resolve_dynamic_batch,
    resolve_dynamic_sequence,
    extract_rnn_states,
    estimate_memory_consumption,
    plan_tensor_lifecycles,
    partition_for_multi_device,
)
from onnx9000.utils.visualize import to_dot


def test_dce():
    """test_dce docstring."""
    g = Graph("test")
    g.outputs.append("out")
    # Live
    g.add_node(Node("Relu", ["in"], ["out"], {}, name="n1"))
    # Dead
    g.add_node(Node("Relu", ["in"], ["dead_out"], {}, name="n2"))

    dead_code_elimination(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].name == "n1"


def test_fusion_and_stubs():
    """test_fusion_and_stubs docstring."""
    g = Graph("test")
    # Just run them to check no crashes
    constant_folding(g)
    fuse_linear_activation(g)
    fuse_consecutive_transpose(g)
    fuse_matmul_add(g)
    enforce_opset_18(g)
    apply_opset_fallbacks(g)
    detect_cycles(g)
    flatten_subgraphs(g)
    optimize_broadcasting(g)
    transform_nchw_to_nhwc(g)
    transform_nhwc_to_nchw(g)
    inject_probes(g, [])
    optimize_for_webgpu(g)
    polyfill_webgpu_unsupported(g)
    insert_qat_nodes(g)
    convert_to_int8(g)
    resolve_dynamic_batch(g)
    resolve_dynamic_sequence(g)
    extract_rnn_states(g)

    mem = estimate_memory_consumption(g)
    assert mem == {}

    lifecycles = plan_tensor_lifecycles(g)
    assert lifecycles == {}

    parts = partition_for_multi_device(g)
    assert "device_0" in parts

    dot = to_dot(g)
    assert "digraph test {" in dot


def test_fusion_real():
    """test_fusion_real docstring."""
    from onnx9000.passes.fusion import fuse_consecutive_transpose, fuse_matmul_add

    g = Graph("test")
    n1 = Node("Transpose", ["a"], ["b"], {"perm": [1, 0]}, name="t1")
    n2 = Node("Transpose", ["b"], ["c"], {"perm": [1, 0]}, name="t2")
    n3 = Node("MatMul", ["c", "d"], ["e"], {}, name="m1")
    n4 = Node("Add", ["e", "f"], ["g"], {}, name="a1")

    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)

    fuse_consecutive_transpose(g)
    # The naive transpose fusion just removes them and maps outputs.
    # The current naive implementation skips if the output doesn't match next input.
    assert len(g.nodes) == 2

    fuse_matmul_add(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "Gemm"


def test_visualize_real():
    """test_visualize_real docstring."""
    g = Graph("test")
    n1 = Node("Relu", ["a"], ["b"], {}, name="r1")
    g.add_node(n1)
    dot = to_dot(g)
    assert "r1" in dot
    assert "a" in dot
    assert "b" in dot
