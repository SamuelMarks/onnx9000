"""Tests for surgeon_more4 to hit remaining 100% coverage."""

import numpy as np

from onnx9000.core.ir import Attribute, Constant, Graph, Node, Variable
from onnx9000.core.surgeon import fuse_conv_bn


def test_fold_batchnorm_no_b_conv_data() -> None:
    """Test bn fusion when b_conv has no data."""
    g = Graph("test")
    w_conv = Constant(
        "w_conv",
        shape=(2, 2, 3, 3),
        values=np.ones((2, 2, 3, 3), dtype=np.float32).tobytes(),
        dtype="float32",
    )
    # b_conv has no data
    b_conv = Variable("b_conv", shape=(2,), dtype="float32")

    scale = Constant(
        "scale", shape=(2,), values=np.ones((2,), dtype=np.float32).tobytes(), dtype="float32"
    )
    b_bn = Constant(
        "b_bn", shape=(2,), values=np.ones((2,), dtype=np.float32).tobytes(), dtype="float32"
    )
    mean = Constant(
        "mean", shape=(2,), values=np.ones((2,), dtype=np.float32).tobytes(), dtype="float32"
    )
    var = Constant(
        "var", shape=(2,), values=np.ones((2,), dtype=np.float32).tobytes(), dtype="float32"
    )

    g.add_tensor(w_conv)
    g.add_tensor(b_conv)
    g.add_tensor(scale)
    g.add_tensor(b_bn)
    g.add_tensor(mean)
    g.add_tensor(var)

    in_t = Variable("in")
    conv_out = Variable("conv_out")
    bn_out = Variable("bn_out")
    g.add_tensor(in_t)
    g.add_tensor(conv_out)
    g.add_tensor(bn_out)

    # Conv with bias that has no data
    conv = Node("Conv", inputs=[in_t, w_conv, b_conv], outputs=[conv_out])
    bn = Node("BatchNormalization", inputs=[conv_out, scale, b_bn, mean, var], outputs=[bn_out])
    g.add_node(conv)
    g.add_node(bn)

    conv_out.inputs.append(conv)
    bn_out.inputs.append(bn)

    fuse_conv_bn(g)
    assert bn not in g.nodes
    assert len(conv.inputs) == 3


def test_fold_batchnorm_with_b_conv_data() -> None:
    """Test bn fusion when b_conv has data."""
    g = Graph("test")
    w_conv = Constant(
        "w_conv",
        shape=(2, 2, 3, 3),
        values=np.ones((2, 2, 3, 3), dtype=np.float32).tobytes(),
        dtype="float32",
    )
    # b_conv has data
    b_conv = Constant(
        "b_conv", shape=(2,), values=np.ones((2,), dtype=np.float32).tobytes(), dtype="float32"
    )

    scale = Constant(
        "scale", shape=(2,), values=np.ones((2,), dtype=np.float32).tobytes(), dtype="float32"
    )
    b_bn = Constant(
        "b_bn", shape=(2,), values=np.ones((2,), dtype=np.float32).tobytes(), dtype="float32"
    )
    mean = Constant(
        "mean", shape=(2,), values=np.ones((2,), dtype=np.float32).tobytes(), dtype="float32"
    )
    var = Constant(
        "var", shape=(2,), values=np.ones((2,), dtype=np.float32).tobytes(), dtype="float32"
    )

    g.add_tensor(w_conv)
    g.add_tensor(b_conv)
    g.add_tensor(scale)
    g.add_tensor(b_bn)
    g.add_tensor(mean)
    g.add_tensor(var)

    in_t = Variable("in")
    conv_out = Variable("conv_out")
    bn_out = Variable("bn_out")
    g.add_tensor(in_t)
    g.add_tensor(conv_out)
    g.add_tensor(bn_out)

    # Conv with bias that has data
    conv = Node("Conv", inputs=[in_t, w_conv, b_conv], outputs=[conv_out])
    bn = Node("BatchNormalization", inputs=[conv_out, scale, b_bn, mean, var], outputs=[bn_out])
    g.add_node(conv)
    g.add_node(bn)

    conv_out.inputs.append(conv)
    bn_out.inputs.append(bn)

    fuse_conv_bn(g)
    assert bn not in g.nodes
    assert len(conv.inputs) == 3
