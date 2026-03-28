"""Test layout optimizer."""

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.hardware.layout import LayoutOptimizer


def create_mock_graph():
    """Test the create_mock_graph functionality."""
    g = Graph("mock_graph")
    t1 = Tensor("input", (1, 3, 224, 224), DType.FLOAT32)
    t2 = Tensor("output", (1, 16, 112, 112), DType.FLOAT32)
    t3 = Tensor(
        "weights",
        (16, 3, 3, 3),
        DType.FLOAT32,
        is_initializer=True,
        data=np.ones((16, 3, 3, 3), dtype=np.float32),
    )
    g.add_tensor(t1)
    g.add_tensor(t2)
    g.add_tensor(t3)
    node = Node("Conv", ["input", "weights"], ["output"], {"layout": "NCHW"})
    g.add_node(node)
    return g


def test_supports_layout_attribute() -> None:
    """Tests the test_supports_layout_attribute functionality."""
    assert LayoutOptimizer.supports_layout_attribute(Node("Conv", [], [], {}))
    assert not LayoutOptimizer.supports_layout_attribute(Node("Relu", [], [], {}))


def test_nchw_to_nhwc_pass() -> None:
    """Tests the test_nchw_to_nhwc_pass functionality."""
    g = create_mock_graph()
    g2 = LayoutOptimizer.nchw_to_nhwc_pass(g)
    assert g2.nodes[0].attributes["layout"] == "NHWC"


def test_nhwc_to_nchw_pass() -> None:
    """Tests the test_nhwc_to_nchw_pass functionality."""
    g = create_mock_graph()
    g.nodes[0].attributes["layout"] = "NHWC"
    g2 = LayoutOptimizer.nhwc_to_nchw_pass(g)
    assert g2.nodes[0].attributes["layout"] == "NCHW"


def test_inject_transposes_for_layout() -> None:
    """Tests the test_inject_transposes_for_layout functionality."""
    g = create_mock_graph()
    g.add_node(Node("Relu", ["output"], ["relu_out"], {}))
    g2 = LayoutOptimizer.inject_transposes_for_layout(g, target_layout="NHWC")
    transposes = [n for n in g2.nodes if n.op_type == "Transpose"]
    assert len(transposes) == 3
    conv_node = [n for n in g2.nodes if n.op_type == "Conv"][0]
    assert conv_node.attributes["layout"] == "NHWC"


def test_transpose_cancellation_pass() -> None:
    """Tests the test_transpose_cancellation_pass functionality."""
    g = Graph("mock_transpose")
    t_node1 = Node("Transpose", ["in"], ["temp"], {"perm": [0, 2, 3, 1]})
    t_node2 = Node("Transpose", ["temp"], ["out"], {"perm": [0, 3, 1, 2]})
    g.add_node(t_node1)
    g.add_node(t_node2)
    g2 = LayoutOptimizer.transpose_cancellation_pass(g)
    ident_node = [n for n in g2.nodes if n.op_type == "Identity"]
    assert len(ident_node) == 1
    assert ident_node[0].inputs == ["in"]
    assert ident_node[0].outputs == ["out"]


def test_transpose_cancellation_pass_no_match() -> None:
    """Tests the test_transpose_cancellation_pass_no_match functionality."""
    g = Graph("mock_transpose")
    t_node1 = Node("Transpose", ["in"], ["temp"], {"perm": [0, 2, 3, 1]})
    t_node2 = Node("Transpose", ["temp"], ["out"], {"perm": [0, 1, 2, 3]})
    g.add_node(t_node1)
    g.add_node(t_node2)
    g2 = LayoutOptimizer.transpose_cancellation_pass(g)
    ident_node = [n for n in g2.nodes if n.op_type == "Identity"]
    assert len(ident_node) == 0


def test_push_transposes_down() -> None:
    """Tests the test_push_transposes_down functionality."""
    g = create_mock_graph()
    g2 = LayoutOptimizer.push_transposes_down(g)
    assert len(g2.nodes) == len(g.nodes)


def test_optimal_layouts() -> None:
    """Tests the test_optimal_layouts functionality."""
    assert LayoutOptimizer.optimal_webgpu_layout() == "NHWC"
    assert LayoutOptimizer.optimal_wasm_simd_layout() == "NCHW"
    assert LayoutOptimizer.optimal_ios_coreml_layout() == "NCHW"
    assert LayoutOptimizer.optimal_android_nnapi_layout() == "NHWC"


def test_pad_to_alignment() -> None:
    """Tests the test_pad_to_alignment functionality."""
    assert LayoutOptimizer.pad_to_alignment((1, 3, 224, 224), 4) == (1, 3, 224, 224)
    assert LayoutOptimizer.pad_to_alignment((1, 3, 224, 223), 4) == (1, 3, 224, 224)
    assert LayoutOptimizer.pad_to_alignment((), 4) == ()
    assert LayoutOptimizer.pad_to_alignment(("dynamic",), 4) == ("dynamic",)


def test_align_tensor_shapes_pass() -> None:
    """Tests the test_align_tensor_shapes_pass functionality."""
    g = create_mock_graph()
    g.tensors["output"].shape = (1, 16, 112, 113)
    g2 = LayoutOptimizer.align_tensor_shapes_pass(g, alignment=4)
    assert g2.tensors["output"].shape == (1, 16, 112, 116)


def test_update_parameters_for_alignment() -> None:
    """Tests the test_update_parameters_for_alignment functionality."""
    node = Node("Conv", [], [], {})
    n2 = LayoutOptimizer.update_parameters_for_alignment(node, 2)
    assert n2.attributes["aligned"] is True


def test_unfold_constants() -> None:
    """Tests the test_unfold_constants functionality."""
    g = create_mock_graph()
    g2 = LayoutOptimizer.unfold_constants(g)
    assert g2.tensors["weights"].shape == (432,)


def test_estimate_vram_usage() -> None:
    """Tests the test_estimate_vram_usage functionality."""
    g = create_mock_graph()
    vram = LayoutOptimizer.estimate_vram_usage(g)
    assert vram == 1406656


def test_chunk_large_tensors_pass() -> None:
    """Tests the test_chunk_large_tensors_pass functionality."""
    g = create_mock_graph()
    g2 = LayoutOptimizer.chunk_large_tensors_pass(g)
    assert len(g2.nodes) == len(g.nodes)


def test_push_transposes_down_real() -> None:
    """Tests the test_push_transposes_down_real functionality."""
    g = Graph("mock")
    g.inputs = ["in"]
    g.outputs = ["out"]
    t1 = Node("Transpose", ["in"], ["t_out"], {"perm": [0, 2, 3, 1]}, "t1")
    n1 = Node("Relu", ["t_out"], ["out"], {}, "r1")
    g.add_node(t1)
    g.add_node(n1)
    g2 = LayoutOptimizer.push_transposes_down(g)
    assert len(g2.nodes) == 2
    assert g2.nodes[0].op_type == "Relu"
    assert g2.nodes[1].op_type == "Transpose"


def test_update_parameters_for_alignment_reshape() -> None:
    """Tests the test_update_parameters_for_alignment_reshape functionality."""
    n = Node("Reshape", ["in"], ["out"], {"shape": [1, 2, 3]}, "r1")
    n2 = LayoutOptimizer.update_parameters_for_alignment(n, 2)
    assert n2.attributes["shape"] == [1, 2, 5]


def test_chunk_large_tensors_pass_real() -> None:
    """Tests the test_chunk_large_tensors_pass_real functionality."""
    g = Graph("mock")
    t = Tensor("large", (10, 100), DType.FLOAT32)
    g.add_tensor(t)
    n = Node("Relu", ["large"], ["out"], {}, "r1")
    g.add_node(n)
    g2 = LayoutOptimizer.chunk_large_tensors_pass(g, max_size=100)
    has_concat = any(node.op_type == "Concat" for node in g2.nodes)
    assert has_concat


def test_push_transposes_down_mixed_inputs() -> None:
    """Tests the test_push_transposes_down_mixed_inputs functionality."""
    g = Graph("mock")
    g.inputs = ["in1", "in2"]
    g.outputs = ["out"]
    t1 = Node("Transpose", ["in1"], ["t_out"], {"perm": [0, 2, 3, 1]}, "t1")
    n1 = Node("Add", ["t_out", "in2"], ["out"], {}, "r1")
    g.add_node(t1)
    g.add_node(n1)
    g2 = LayoutOptimizer.push_transposes_down(g)
    assert len(g2.nodes) == 2
