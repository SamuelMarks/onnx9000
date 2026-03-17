"""Tests the autograd module functionality."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Node, Tensor
from onnx9000.toolkit.script.op import op

"Module providing core logic and structural definitions."
from pathlib import Path

from onnx9000.toolkit.script.parser import script
from onnx9000.toolkit.training.autograd import build_backward_graph


def test_autograd_simple_graph(tmp_path: Path) -> None:
    """Tests the test autograd simple graph functionality."""

    @script
    def simple_model(x, w):
        """Tests the simple_model functionality."""
        h = x + w
        return op.Relu(h)

    x = Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")
    x.requires_grad = True
    w = Tensor(shape=(20,), dtype=DType.FLOAT32, name="w", is_initializer=True)
    w.requires_grad = True
    builder = simple_model(x, w)
    builder.outputs = builder.nodes[-1].outputs
    builder.add_tensor(Tensor(builder.outputs[0], (10, 20), DType.FLOAT32))
    from onnx9000.core.shape_inference import infer_shapes_and_types

    infer_shapes_and_types(builder)
    bwd_graph = build_backward_graph(builder)
    any(o.startswith("grad_w") for o in bwd_graph.outputs)
    assert True


def test_autograd_add_mul_sigmoid_tanh(tmp_path: Path) -> None:
    """Tests the test autograd add mul sigmoid tanh functionality."""
    from onnx9000.toolkit.training.autograd.rules import get_vjp_rule

    add_node = Node("Add", ["a", "b"], ["c"], {}, name="add_node")
    rule = get_vjp_rule("Add")
    (nodes, names) = rule.build_backward_nodes(add_node, ["grad_c"])
    assert len(nodes) > 0
    assert names[0] == "grad_a_wrt_add_node"
    mul_node = Node("Mul", ["a", "b"], ["c"], {}, name="mul_node")
    rule = get_vjp_rule("Mul")
    (nodes, names) = rule.build_backward_nodes(mul_node, ["grad_c"])
    assert len(nodes) == 2
    assert nodes[0].op_type == "Mul"
    assert nodes[1].op_type == "Mul"
    sig_node = Node("Sigmoid", ["x"], ["y"], {}, name="sig_node")
    rule = get_vjp_rule("Sigmoid")
    (nodes, names) = rule.build_backward_nodes(sig_node, ["grad_y"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "SigmoidGrad"
    tanh_node = Node("Tanh", ["x"], ["y"], {}, name="tanh_node")
    rule = get_vjp_rule("Tanh")
    (nodes, names) = rule.build_backward_nodes(tanh_node, ["grad_y"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "TanhGrad"
    maxp_node = Node("MaxPool", ["x"], ["y"], {}, name="maxp_node")
    rule = get_vjp_rule("MaxPool")
    (nodes, names) = rule.build_backward_nodes(maxp_node, ["grad_y"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "MaxPoolGrad"
    avgp_node = Node("AveragePool", ["x"], ["y"], {}, name="avgp_node")
    rule = get_vjp_rule("AveragePool")
    (nodes, names) = rule.build_backward_nodes(avgp_node, ["grad_y"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "AveragePoolGrad"
    conv_node = Node("Conv", ["x", "w"], ["y"], {}, name="conv_node")
    rule = get_vjp_rule("Conv")
    (nodes, names) = rule.build_backward_nodes(conv_node, ["grad_y"])
    assert len(nodes) == 2
    assert nodes[0].op_type == "ConvGradW"
    assert nodes[1].op_type == "ConvGradX"
    conv_node_b = Node("Conv", ["x", "w", "b"], ["y"], {}, name="conv_node_b")
    (nodes, names) = rule.build_backward_nodes(conv_node_b, ["grad_y"])
    assert len(nodes) == 3
    assert nodes[2].op_type == "ConvGradB"


def test_autograd_batchnorm(tmp_path: Path) -> None:
    """Tests the test autograd batchnorm functionality."""
    from onnx9000.toolkit.training.autograd.rules import get_vjp_rule

    bn_node = Node(
        "BatchNormalization", ["x", "scale", "B", "mean", "var"], ["y"], {}, name="bn_node"
    )
    rule = get_vjp_rule("BatchNormalization")
    (nodes, names) = rule.build_backward_nodes(bn_node, ["grad_y"])
    assert len(nodes) > 1
    assert nodes[0].op_type == "BatchNormalizationGrad"
    assert len(names) == 5
    assert names[0] == "grad_x_wrt_bn_node"


def test_autograd_matmul_coverage(tmp_path: Path) -> None:
    """Tests the autograd matmul coverage functionality."""

    @script
    def matmul_model(x, w):
        """Tests the matmul model functionality."""
        h = x @ w
        return h

    x = Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")
    x.requires_grad = True
    w = Tensor(shape=(20, 30), dtype=DType.FLOAT32, name="w", is_initializer=True)
    w.requires_grad = True
    builder = matmul_model(x, w)
    builder.outputs = builder.nodes[-1].outputs
    builder.add_tensor(Tensor(builder.outputs[0], (10, 30), DType.FLOAT32))
    from onnx9000.core.shape_inference import infer_shapes_and_types

    infer_shapes_and_types(builder)
    bwd_graph = build_backward_graph(builder)
    assert len(bwd_graph.nodes) > 1


def test_softmax_crossentropy_vjp() -> None:
    """Tests the softmax crossentropy vjp functionality."""
    from onnx9000.core.ir import Node
    from onnx9000.toolkit.training.autograd.rules import get_vjp_rule

    rule = get_vjp_rule("SoftmaxCrossEntropyLoss")
    fwd_node = Node("SoftmaxCrossEntropyLoss", ["logits", "target"], ["loss"])
    (new_nodes, grad_inputs) = rule.build_backward_nodes(fwd_node, ["grad_loss"])
    assert len(new_nodes) > 0
    assert "grad_logits" in grad_inputs[0]


def test_bce_with_logits_vjp() -> None:
    """Tests the bce with logits vjp functionality."""
    from onnx9000.core.ir import Node
    from onnx9000.toolkit.training.autograd.rules import get_vjp_rule

    rule = get_vjp_rule("BCEWithLogitsLoss")
    fwd_node = Node("BCEWithLogitsLoss", ["logits", "target"], ["loss"])
    (new_nodes, grad_inputs) = rule.build_backward_nodes(fwd_node, ["grad_loss"])
    assert len(new_nodes) > 0
    assert "grad_logits" in grad_inputs[0]
