"""Exhaustive unit tests for VJP differentiation rules."""

from onnx9000.core.ir import Node
from onnx9000.toolkit.training.autograd.rules import (
    _VJP_REGISTRY,
    BinaryCrossEntropyLossVJP,
    RecurrentVJP,
    ResizeVJP,
    SequenceConstructVJP,
    SiluVJP,
    SplitToSequenceVJP,
    register_vjp,
)


def test_silu_vjp() -> None:
    """Tests the SiluVJP backward node construction."""
    node = Node("Silu", ["x"], ["y"], {}, name="silu_1")
    vjp = SiluVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dy"])
    assert len(nodes) > 0
    assert len(grads) == 1


def test_resize_vjp() -> None:
    """Tests the ResizeVJP backward node construction."""
    node = Node("Resize", ["x", "roi", "scales", "sizes"], ["y"], {}, name="resize_1")
    vjp = ResizeVJP()
    assert vjp.build_backward_nodes(node, ["dy"]) is None

    node_cubic = Node(
        "Resize", ["x", "roi", "scales", "sizes"], ["y"], {"mode": "cubic"}, name="resize_c"
    )
    nodes, grads = vjp.build_backward_nodes(node_cubic, ["dy"])
    assert len(nodes) == 1
    assert len(grads) == 4


def test_split_to_sequence_vjp() -> None:
    """Tests the SplitToSequenceVJP backward node construction with split input."""
    node = Node("SplitToSequence", ["seq", "split"], ["y"], {"axis": 1}, name="sts_1")
    vjp = SplitToSequenceVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dy"])
    assert len(nodes) == 2
    assert nodes[0].op_type == "ConcatFromSequence"
    assert len(grads) == 2


def test_split_to_sequence_vjp_no_split() -> None:
    """Tests the SplitToSequenceVJP backward node construction without split input."""
    node = Node("SplitToSequence", ["seq"], ["y"], {"axis": 1}, name="sts_1")
    vjp = SplitToSequenceVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dy"])
    assert len(nodes) == 1
    assert len(grads) == 1


def test_binary_cross_entropy_loss_vjp() -> None:
    """Tests the BinaryCrossEntropyLossVJP backward node construction."""
    node = Node(
        "BinaryCrossEntropyLoss",
        ["pred", "target", "weight"],
        ["loss"],
        {},
        name="bce_1",
    )
    vjp = BinaryCrossEntropyLossVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dloss"])
    assert len(nodes) > 0
    assert len(grads) == 2


def test_sequence_construct_vjp() -> None:
    """Tests the SequenceConstructVJP backward node construction."""
    node = Node("SequenceConstruct", ["a", "b", "c"], ["seq"], {}, name="sc_1")
    vjp = SequenceConstructVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dseq"])
    assert len(nodes) == 6
    assert len(grads) == 3


def test_recurrent_vjp() -> None:
    """Tests the RecurrentVJP backward node construction."""
    node = Node("Recurrent", ["M", "cond", "v_initial"], ["v_final"], {}, name="loop_1")
    vjp = RecurrentVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dv_final"])
    assert len(nodes) == 1
    assert len(grads) == 3


def test_register_custom_vjp() -> None:
    """Tests custom VJP registration."""

    def custom_vjp_function(
        fwd_node: Node, grad_outputs: list[str]
    ) -> tuple[list[Node], list[str]]:
        """Execute the custom vjp function operation."""
        return [], ["grad_x"]

    register_vjp("MyCustomOp", custom_vjp_function)

    assert "MyCustomOp" in _VJP_REGISTRY
    vjp_inst = _VJP_REGISTRY["MyCustomOp"]
    node = Node("MyCustomOp", ["x"], ["y"])
    nodes, grads = vjp_inst.build_backward_nodes(node, ["dy"])
    assert nodes == []
    assert grads == ["grad_x"]


def test_resize_vjp_bilinear() -> None:
    """Tests the ResizeVJP bilinear mode backward node construction."""
    node = Node(
        "Resize",
        ["x", "roi", "scales", "sizes"],
        ["y"],
        {"mode": "bilinear"},
        name="resize_2",
    )
    vjp = ResizeVJP()
    assert vjp.build_backward_nodes(node, ["dy"]) is None
