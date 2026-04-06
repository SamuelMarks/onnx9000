"""Tests for surgeon_more3 to hit 100% coverage."""

import numpy as np
from onnx9000.core.ir import Attribute, Constant, Graph, Node, Variable
from onnx9000.core.surgeon import fuse_conv_bn, map_alibi, unroll_scan


def test_fold_batchnorm_fallback() -> None:
    """Test fallback to structural fusion if data isn't available."""
    g = Graph("test")
    w_conv = Variable("w_conv", shape=(64, 3, 3, 3))
    b_conv = Variable("b_conv", shape=(64,))
    scale = Variable("scale", shape=(64,))
    b_bn = Variable("b_bn", shape=(64,))
    mean = Variable("mean", shape=(64,))
    var = Variable("var", shape=(64,))

    # Missing data inside scale/mean/var triggers fallback
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

    conv = Node("Conv", inputs=[in_t, w_conv, b_conv], outputs=[conv_out])
    bn = Node("BatchNormalization", inputs=[conv_out, scale, b_bn, mean, var], outputs=[bn_out])
    g.add_node(conv)
    g.add_node(bn)

    conv_out.inputs.append(conv)
    bn_out.inputs.append(bn)

    fuse_conv_bn(g)
    assert bn not in g.nodes
    assert conv.outputs[0] == bn_out


def test_fold_batchnorm_no_b_conv() -> None:
    """Test bn fusion when conv has no bias."""
    g = Graph("test")
    w_conv = Constant(
        "w_conv",
        shape=(2, 2, 3, 3),
        values=np.ones((2, 2, 3, 3), dtype=np.float32).tobytes(),
        dtype="float32",
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

    # Conv with no bias
    conv = Node("Conv", inputs=[in_t, w_conv], outputs=[conv_out])
    bn = Node("BatchNormalization", inputs=[conv_out, scale, b_bn, mean, var], outputs=[bn_out])
    g.add_node(conv)
    g.add_node(bn)

    conv_out.inputs.append(conv)
    bn_out.inputs.append(bn)

    fuse_conv_bn(g)
    assert bn not in g.nodes
    assert len(conv.inputs) == 3


def test_fold_batchnorm_exception() -> None:
    """Test bn fusion when exception occurs."""
    g = Graph("test")
    # Wrong shape to trigger exception during reshaping
    w_conv = Constant("w_conv", shape=(2, 2, 3, 3), values=b"bad", dtype="float32")
    scale = Constant(
        "scale", shape=(1,), values=np.ones((1,), dtype=np.float32).tobytes(), dtype="float32"
    )
    b_bn = Constant(
        "b_bn", shape=(1,), values=np.ones((1,), dtype=np.float32).tobytes(), dtype="float32"
    )
    mean = Constant(
        "mean", shape=(1,), values=np.ones((1,), dtype=np.float32).tobytes(), dtype="float32"
    )
    var = Constant(
        "var", shape=(1,), values=np.ones((1,), dtype=np.float32).tobytes(), dtype="float32"
    )

    g.add_tensor(w_conv)
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

    conv = Node("Conv", inputs=[in_t, w_conv], outputs=[conv_out])
    bn = Node("BatchNormalization", inputs=[conv_out, scale, b_bn, mean, var], outputs=[bn_out])
    g.add_node(conv)
    g.add_node(bn)

    conv_out.inputs.append(conv)
    bn_out.inputs.append(bn)

    fuse_conv_bn(g)
    assert bn in g.nodes


def test_map_alibi() -> None:
    """Test map_alibi else block."""
    g = Graph("test")
    # Not an alibi
    n = Node("Add", inputs=["a", "b"], outputs=["c"])
    g.add_node(n)
    map_alibi(g)
    assert g.nodes[0] == n


def test_unroll_scan() -> None:
    """Test unroll_scan else blocks."""
    g = Graph("test")
    n = Node("Scan", inputs=["a"], outputs=["b"])
    # Not jax.lax.scan
    g.add_node(n)

    unroll_scan(g)
    assert g.nodes[0] == n

    n.op_type = "jax.lax.scan"
    n.attributes["sequence_length"] = Attribute("sequence_length", value=200)
    unroll_scan(g)
    assert g.nodes[0] == n
