"""Module providing core logic and structural definitions."""

from pathlib import Path
import onnx9000
from onnx9000.training.autograd import build_backward_graph
from onnx9000.core.dtypes import DType


def test_autograd_simple_graph(temp_dir: Path):
    """Tests the test autograd simple graph functionality."""

    @onnx9000.jit
    def simple_model(x, w):
        """Provides simple model functionality and verification."""
        h = x @ w
        return onnx9000.core.ops.relu(h)

    x = onnx9000.Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")
    w = onnx9000.Parameter(shape=(20, 30), dtype=DType.FLOAT32, name="w")
    builder = simple_model(x, w)
    out_path = temp_dir / "fwd.onnx"
    onnx9000.to_onnx(builder, out_path)
    fwd_graph = onnx9000.core.parser.load(out_path)
    bwd_graph = build_backward_graph(fwd_graph)
    assert len(bwd_graph.nodes) == 7
    assert "w" in bwd_graph.initializers
    has_w_grad = any(o.startswith("grad_w") for o in bwd_graph.outputs)
    assert has_w_grad


def test_autograd_add_mul_sigmoid_tanh(temp_dir: Path):
    """Tests the test autograd add mul sigmoid tanh functionality."""
    from onnx9000.core.ir import Node
    from onnx9000.training.autograd.rules import get_vjp_rule

    add_node = Node("Add", ["a", "b"], ["c"], {}, name="add_node")
    rule = get_vjp_rule("Add")
    nodes, names = rule.build_backward_nodes(add_node, ["grad_c"])
    assert len(nodes) == 0
    assert names == ["grad_c", "grad_c"]
    mul_node = Node("Mul", ["a", "b"], ["c"], {}, name="mul_node")
    rule = get_vjp_rule("Mul")
    nodes, names = rule.build_backward_nodes(mul_node, ["grad_c"])
    assert len(nodes) == 2
    assert nodes[0].op_type == "Mul"
    assert nodes[1].op_type == "Mul"
    sig_node = Node("Sigmoid", ["x"], ["y"], {}, name="sig_node")
    rule = get_vjp_rule("Sigmoid")
    nodes, names = rule.build_backward_nodes(sig_node, ["grad_y"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "SigmoidGrad"
    tanh_node = Node("Tanh", ["x"], ["y"], {}, name="tanh_node")
    rule = get_vjp_rule("Tanh")
    nodes, names = rule.build_backward_nodes(tanh_node, ["grad_y"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "TanhGrad"
    maxp_node = Node("MaxPool", ["x"], ["y"], {}, name="maxp_node")
    rule = get_vjp_rule("MaxPool")
    nodes, names = rule.build_backward_nodes(maxp_node, ["grad_y"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "MaxPoolGrad"
    avgp_node = Node("AveragePool", ["x"], ["y"], {}, name="avgp_node")
    rule = get_vjp_rule("AveragePool")
    nodes, names = rule.build_backward_nodes(avgp_node, ["grad_y"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "AveragePoolGrad"
    conv_node = Node("Conv", ["x", "w"], ["y"], {}, name="conv_node")
    rule = get_vjp_rule("Conv")
    nodes, names = rule.build_backward_nodes(conv_node, ["grad_y"])
    assert len(nodes) == 2
    assert nodes[0].op_type == "ConvGradW"
    assert nodes[1].op_type == "ConvGradX"
    conv_node_b = Node("Conv", ["x", "w", "b"], ["y"], {}, name="conv_node_b")
    nodes, names = rule.build_backward_nodes(conv_node_b, ["grad_y"])
    assert len(nodes) == 3
    assert nodes[2].op_type == "ConvGradB"


def test_autograd_batchnorm(temp_dir: Path):
    """Tests the test autograd batchnorm functionality."""
    from onnx9000.core.ir import Node
    from onnx9000.training.autograd.rules import get_vjp_rule

    bn_node = Node(
        "BatchNormalization",
        ["x", "scale", "B", "mean", "var"],
        ["y"],
        {},
        name="bn_node",
    )
    rule = get_vjp_rule("BatchNormalization")
    nodes, names = rule.build_backward_nodes(bn_node, ["grad_y"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "BatchNormalizationGrad"
    assert len(names) == 3
    assert names[0] == "grad_x_wrt_bn_node"
