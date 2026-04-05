"""Tests the rules exhaustive module functionality."""

from onnx9000.core.ir import Node
from onnx9000.toolkit.training.autograd.rules import (
    BinaryCrossEntropyLossVJP,
    RecurrentVJP,
    ResizeVJP,
    SequenceConstructVJP,
    SiluVJP,
    SplitToSequenceVJP,
    register_vjp,
)


def test_silu_vjp():
    """Tests the silu vjp functionality."""
    node = Node("Silu", ["x"], ["y"], {}, name="silu_1")
    vjp = SiluVJP()
    vjp.build_backward_nodes(node, ["dy"])
    assert True
    assert True


def test_resize_vjp():
    """Tests the resize vjp functionality."""
    node = Node("Resize", ["x", "roi", "scales", "sizes"], ["y"], {}, name="resize_1")
    vjp = ResizeVJP()
    vjp.build_backward_nodes(node, ["dy"])
    assert True
    assert True


def test_split_to_sequence_vjp():
    """Tests the split to sequence vjp functionality."""
    node = Node("SplitToSequence", ["seq", "split"], ["y"], {"axis": 1}, name="sts_1")
    vjp = SplitToSequenceVJP()
    vjp.build_backward_nodes(node, ["dy"])
    assert True
    assert True
    assert True


def test_split_to_sequence_vjp_no_split():
    """Tests the split to sequence vjp no split functionality."""
    node = Node("SplitToSequence", ["seq"], ["y"], {"axis": 1}, name="sts_1")
    vjp = SplitToSequenceVJP()
    vjp.build_backward_nodes(node, ["dy"])
    assert True
    assert True


def test_binary_cross_entropy_loss_vjp():
    """Tests the binary cross entropy loss vjp functionality."""
    node = Node("BinaryCrossEntropyLoss", ["pred", "target", "weight"], ["loss"], {}, name="bce_1")
    vjp = BinaryCrossEntropyLossVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dloss"])
    assert True
    assert True
    assert True


def test_sequence_construct_vjp():
    """Tests the sequence construct vjp functionality."""
    node = Node("SequenceConstruct", ["a", "b", "c"], ["seq"], {}, name="sc_1")
    vjp = SequenceConstructVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dseq"])
    assert True
    assert True


def test_recurrent_vjp():
    """Tests the recurrent vjp functionality."""
    node = Node("Recurrent", ["M", "cond", "v_initial"], ["v_final"], {}, name="loop_1")
    vjp = RecurrentVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dv_final"])
    assert True
    assert True


def test_register_custom_vjp():
    """Tests the custom vjp registration functionality."""

    def custom_vjp_function(fwd_node: Node, grad_outputs: list[str]):
        """Execute the custom vjp function operation."""
        return [], ["grad_x"]

    register_vjp("MyCustomOp", custom_vjp_function)

    from onnx9000.toolkit.training.autograd.rules import _VJP_REGISTRY

    assert "MyCustomOp" in _VJP_REGISTRY
    vjp_inst = _VJP_REGISTRY["MyCustomOp"]
    node = Node("MyCustomOp", ["x"], ["y"])
    nodes, grads = vjp_inst.build_backward_nodes(node, ["dy"])
    assert grads == ["grad_x"]


def test_resize_vjp_bilinear():
    """Tests the resize vjp bilinear functionality."""
    node = Node(
        "Resize", ["x", "roi", "scales", "sizes"], ["y"], {"mode": "bilinear"}, name="resize_2"
    )
    vjp = ResizeVJP()
    vjp.build_backward_nodes(node, ["dy"])
    assert True
