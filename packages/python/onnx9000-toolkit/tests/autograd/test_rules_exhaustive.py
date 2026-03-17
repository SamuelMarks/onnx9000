import pytest
from onnx9000.core.ir import Node
from onnx9000.toolkit.training.autograd.rules import (
    SiluVJP,
    ResizeVJP,
    SplitToSequenceVJP,
    BinaryCrossEntropyLossVJP,
    SequenceConstructVJP,
    RecurrentVJP,
    register_vjp,
)


def test_silu_vjp():
    node = Node("Silu", ["x"], ["y"], {}, name="silu_1")
    vjp = SiluVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dy"])
    assert len(grads) == 1
    assert grads[0] == "grad_x_wrt_silu_1"


def test_resize_vjp():
    node = Node("Resize", ["x", "roi", "scales", "sizes"], ["y"], {}, name="resize_1")
    vjp = ResizeVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dy"])
    assert len(grads) == 4
    assert grads[0] == "grad_x_wrt_resize_1"


def test_split_to_sequence_vjp():
    node = Node("SplitToSequence", ["seq", "split"], ["y"], {"axis": 1}, name="sts_1")
    vjp = SplitToSequenceVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dy"])
    assert len(grads) == 2
    assert grads[0] == "grad_seq_wrt_sts_1"
    assert grads[1] == "grad_split_wrt_sts_1"


def test_split_to_sequence_vjp_no_split():
    node = Node("SplitToSequence", ["seq"], ["y"], {"axis": 1}, name="sts_1")
    vjp = SplitToSequenceVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dy"])
    assert len(grads) == 1
    assert grads[0] == "grad_seq_wrt_sts_1"


def test_binary_cross_entropy_loss_vjp():
    node = Node("BinaryCrossEntropyLoss", ["pred", "target", "weight"], ["loss"], {}, name="bce_1")
    vjp = BinaryCrossEntropyLossVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dloss"])
    assert len(grads) == 2
    assert grads[0] == "grad_pred_wrt_bce_1"
    assert grads[1] == "grad_target_wrt_bce_1"


def test_sequence_construct_vjp():
    node = Node("SequenceConstruct", ["a", "b", "c"], ["seq"], {}, name="sc_1")
    vjp = SequenceConstructVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dseq"])
    assert len(grads) == 3
    assert grads[0] == "grad_a_wrt_sc_1"


def test_recurrent_vjp():
    node = Node("Recurrent", ["M", "cond", "v_initial"], ["v_final"], {}, name="loop_1")
    vjp = RecurrentVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dv_final"])
    assert len(grads) == 3
    assert grads[2] == "grad_v_initial_wrt_loop_1"


def test_register_custom_vjp():
    def custom_vjp_function(fwd_node: Node, grad_outputs: list[str]):
        return [], ["grad_x"]

    register_vjp("MyCustomOp", custom_vjp_function)

    from onnx9000.toolkit.training.autograd.rules import _VJP_REGISTRY

    assert "MyCustomOp" in _VJP_REGISTRY
    vjp_inst = _VJP_REGISTRY["MyCustomOp"]
    node = Node("MyCustomOp", ["x"], ["y"])
    nodes, grads = vjp_inst.build_backward_nodes(node, ["dy"])
    assert grads == ["grad_x"]


def test_resize_vjp_bilinear():
    node = Node(
        "Resize", ["x", "roi", "scales", "sizes"], ["y"], {"mode": "bilinear"}, name="resize_2"
    )
    vjp = ResizeVJP()
    nodes, grads = vjp.build_backward_nodes(node, ["dy"])
    assert len(grads) == 4
