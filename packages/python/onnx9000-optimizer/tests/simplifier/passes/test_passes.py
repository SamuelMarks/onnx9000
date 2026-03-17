"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.simplifier.passes import (
    apply_opset_fallbacks,
    constant_folding,
    convert_to_int8,
    dead_code_elimination,
    detect_cycles,
    enforce_opset_18,
    estimate_memory_consumption,
    extract_rnn_states,
    flatten_subgraphs,
    fuse_consecutive_transpose,
    fuse_linear_activation,
    fuse_matmul_add,
    inject_probes,
    insert_qat_nodes,
    optimize_broadcasting,
    optimize_for_webgpu,
    partition_for_multi_device,
    plan_tensor_lifecycles,
    polyfill_webgpu_unsupported,
    resolve_dynamic_batch,
    resolve_dynamic_sequence,
    transform_nchw_to_nhwc,
    transform_nhwc_to_nchw,
)


def test_dce() -> None:
    """Tests the test dce functionality."""
    g = Graph("test")
    g.outputs.append("out")
    g.add_node(Node("Relu", ["in"], ["out"], {}, name="n1"))
    g.add_node(Node("Relu", ["in"], ["dead_out"], {}, name="n2"))
    dead_code_elimination(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].name == "n1"


def test_fusion_and_stubs() -> None:
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


def test_fusion_real() -> None:
    """Tests the test fusion real functionality."""
    from onnx9000.optimizer.simplifier.passes.fusion import (
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


def test_visualize_real() -> None:
    """Tests the test visualize real functionality."""
    g = Graph("test")
    n1 = Node("Relu", ["a"], ["b"], {}, name="r1")
    g.add_node(n1)


def test_transpose_identity_fusion_cov() -> None:
    """Tests the transpose identity fusion cov functionality."""
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.optimizer.simplifier.passes.fusion import fuse_consecutive_transpose

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
