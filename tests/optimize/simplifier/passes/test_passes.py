"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Graph, Node
from onnx9000.optimize.simplifier.passes import (
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
    """Tests the test dce functionality."""
    g = Graph("test")
    g.outputs.append("out")
    g.add_node(Node("Relu", ["in"], ["out"], {}, name="n1"))
    g.add_node(Node("Relu", ["in"], ["dead_out"], {}, name="n2"))
    dead_code_elimination(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].name == "n1"


def test_fusion_and_stubs():
    """Tests the test fusion and stubs functionality."""
    g = Graph("test")
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
    inject_probes(g)
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
    """Tests the test fusion real functionality."""
    from onnx9000.optimize.simplifier.passes.fusion import (
        fuse_consecutive_transpose,
        fuse_matmul_add,
    )

    g = Graph("test")
    n1 = Node("Transpose", ["a"], ["b"], {"perm": [1, 0]}, name="t1")
    n2 = Node("Transpose", ["b"], ["c"], {"perm": [1, 0]}, name="t2")
    n3 = Node("MatMul", ["c", "d"], ["e"], {}, name="m1")
    n4 = Node("Add", ["e", "f"], ["g"], {}, name="a1")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)
    g.outputs = ["g"]
    fuse_consecutive_transpose(g)
    assert len(g.nodes) == 2
    fuse_matmul_add(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "Gemm"


def test_visualize_real():
    """Tests the test visualize real functionality."""
    g = Graph("test")
    n1 = Node("Relu", ["a"], ["b"], {}, name="r1")
    g.add_node(n1)
    dot = to_dot(g)
    assert "r1" in dot
    assert "a" in dot
    assert "b" in dot


def test_transpose_identity_fusion_cov():
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.optimize.simplifier.passes.fusion import fuse_consecutive_transpose

    g = Graph("test")
    x = Tensor("x", (2, 3), "float")
    y = Tensor("y", (3, 2), "float")
    z = Tensor("z", (2, 3), "float")
    g.add_tensor(x)
    g.add_tensor(y)
    g.add_tensor(z)
    n1 = Node("Transpose", ["x"], ["y"], {})
    n2 = Node("Transpose", ["y"], ["z"], {})
    g.add_node(n1)
    g.add_node(n2)
    g.outputs.append("z")
    fuse_consecutive_transpose(g)
    assert len(g.nodes) == 0
    assert g.outputs[0] == "x"
