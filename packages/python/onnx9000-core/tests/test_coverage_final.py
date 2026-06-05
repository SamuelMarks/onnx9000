"""Tests for coverage final."""

import ast
import builtins
import struct
from unittest.mock import patch

import numpy as np
import pytest
from onnx9000.core.ir import Attribute, Constant, DType, DynamicDim, Graph, Node, Tensor, Variable
from onnx9000.core.registry import OperatorRegistry, UnsupportedOpError, register_op
from onnx9000.core.surgeon import (
    PatternMatcher,
    convert_layout,
    evaluate_math_graph,
    extract_scalar,
    fold_constants_math,
    fold_constants_shape,
    fuse_conv_bn,
    match_pattern,
    merge_lora_adapters,
    reconstruct_sequences,
    restore_layouts,
    transpose_constant,
)
from onnx9000.core.symbolic import _ast_to_str, simplify_expression


def test_registry_fallback_coverage():
    """Test registry fallback coverage."""
    reg = OperatorRegistry()

    def my_op():
        """My op."""
        return "ok"

    reg.register_op("test", "MyOp")(my_op)
    assert reg.get_op("test", "MyOp", provider="cuda") == my_op
    with pytest.raises(UnsupportedOpError):
        reg.get_op("test", "NonExistent", provider="cuda")


def test_symbolic_ast_to_str_fallback():
    """Test symbolic AST to string fallback."""
    node = ast.UnaryOp(op=ast.USub(), operand=ast.Name(id="x", ctx=ast.Load()))
    s = _ast_to_str(node)
    assert "UnaryOp" in s


def test_match_pattern_recursive_subgraph():
    """Test matching patterns in recursive subgraphs."""
    g = Graph("main")
    sub_g = Graph("sub")
    sub_n = Node("Relu", ["in"], ["out"])
    sub_g.add_node(sub_n)
    n = Node(
        "If",
        ["cond"],
        ["out"],
        attributes={"then_branch": Attribute("then_branch", "GRAPH", sub_g)},
    )
    g.add_node(n)
    pattern = PatternMatcher(op_type="Relu")
    matches = match_pattern(g, pattern, recursive=True)
    assert len(matches) == 1
    assert matches[0].op_type == "Relu"


def test_match_pattern_recursive_list_graphs():
    """Test matching patterns in recursive list of graphs (e.g. Scan)."""
    g = Graph("main")
    sub_g1 = Graph("sub1")
    sub_n1 = Node("Relu", ["in1"], ["out1"])
    sub_g1.add_node(sub_n1)

    sub_g2 = Graph("sub2")
    sub_n2 = Node("Relu", ["in2"], ["out2"])
    sub_g2.add_node(sub_n2)

    # Simulate a Scan-like node with a list of graphs
    n = Node(
        "Scan",
        ["in"],
        ["out"],
        attributes={"body": Attribute("body", "LIST_GRAPH", [sub_g1, sub_g2])},
    )
    g.add_node(n)

    pattern = PatternMatcher(op_type="Relu")
    matches = match_pattern(g, pattern, recursive=True)
    assert len(matches) == 2


def test_fold_constants_math_toposort_exception():
    """Test constant folding with topological sort exception (cycle)."""
    g = Graph("cyclic")
    v1 = Variable("v1")
    v2 = Variable("v2")
    g.add_tensor(v1)
    g.add_tensor(v2)
    n1 = Node("Add", [v1], [v2])
    n2 = Node("Add", [v2], [v1])
    v2.outputs.append(n2)
    v1.outputs.append(n1)
    g.add_node(n1)
    g.add_node(n2)
    fold_constants_math(g)


def test_fold_constants_math_evaluate_exception():
    """Test constant folding with evaluation exception."""
    g = Graph("test")
    c1 = Constant("c1", values=struct.pack("<f", 1.0))
    c2 = Constant("c2", values=struct.pack("<f", 2.0))
    g.add_tensor(c1)
    g.add_tensor(c2)
    n = Node("Add", ["c1", "c2"], ["out"])
    g.add_node(n)

    with patch("onnx9000.core.surgeon.evaluate_math_graph", side_effect=Exception("mock")):
        fold_constants_math(g)


def test_restore_layouts_conv():
    """Test layout restoration for Convolution nodes."""
    g = Graph("test")
    n = Node("Conv", ["in", "w"], ["out"])
    g.add_node(n)
    restore_layouts(g)


def test_fuse_conv_bn_coverage():
    """Test Conv-BN fusion with Tensor output to cover all lines."""
    g = Graph("fuse")
    v_in = Variable("in", [1, 3, 224, 224], DType.FLOAT32)
    v_conv = Variable("v_conv", [1, 16, 224, 224], DType.FLOAT32)
    v_out = Variable("out", [1, 16, 224, 224], DType.FLOAT32)

    w = Constant(
        "w",
        values=np.zeros((16, 3, 3, 3), dtype=np.float32).tobytes(),
        shape=(16, 3, 3, 3),
        dtype=DType.FLOAT32,
    )
    scale = Constant(
        "scale", values=np.ones(16, dtype=np.float32).tobytes(), shape=(16,), dtype=DType.FLOAT32
    )
    b = Constant(
        "b", values=np.zeros(16, dtype=np.float32).tobytes(), shape=(16,), dtype=DType.FLOAT32
    )
    mean = Constant(
        "mean", values=np.zeros(16, dtype=np.float32).tobytes(), shape=(16,), dtype=DType.FLOAT32
    )
    var = Constant(
        "var", values=np.ones(16, dtype=np.float32).tobytes(), shape=(16,), dtype=DType.FLOAT32
    )

    g.add_tensor(v_in)
    g.add_tensor(v_conv)
    g.add_tensor(v_out)
    g.add_tensor(w)
    g.add_tensor(scale)
    g.add_tensor(b)
    g.add_tensor(mean)
    g.add_tensor(var)

    n_conv = Node("Conv", ["in", "w"], [v_conv])
    n_bn = Node("BatchNormalization", [v_conv, "scale", "b", "mean", "var"], [v_out])

    g.add_node(n_conv)
    g.add_node(n_bn)

    fuse_conv_bn(g)
    assert n_bn not in g.nodes
    assert n_conv.outputs[0] == v_out


def test_fuse_conv_bn_invalid_nodes():
    """Test Conv-BN fusion with invalid nodes to cover continue branches."""
    g = Graph("fuse_invalid")
    v_in = Variable("in")
    v_conv = Variable("v_conv")
    v_out = Variable("out")
    g.add_tensor(v_in)
    g.add_tensor(v_conv)
    g.add_tensor(v_out)

    # 1. Conv with 1 input (missing weight)
    n_conv1 = Node("Conv", [v_in], [v_conv])
    n_bn1 = Node("BatchNormalization", [v_conv, "s", "b", "m", "v"], [v_out])
    g.add_node(n_conv1)
    g.add_node(n_bn1)
    fuse_conv_bn(g)
    assert n_bn1 in g.nodes  # Not fused

    # 2. Conv with weight NOT in graph.tensors
    g = Graph("fuse_invalid2")
    g.add_tensor(v_in)
    g.add_tensor(v_conv)
    g.add_tensor(v_out)
    n_conv2 = Node("Conv", [v_in, "missing_w"], [v_conv])
    n_bn2 = Node("BatchNormalization", [v_conv, "s", "b", "m", "v"], [v_out])
    g.add_node(n_conv2)
    g.add_node(n_bn2)
    fuse_conv_bn(g)
    assert n_bn2 in g.nodes

    # 3. BN with < 5 inputs
    g = Graph("fuse_invalid3")
    g.add_tensor(v_in)
    g.add_tensor(v_conv)
    g.add_tensor(v_out)
    w = Constant("w", values=b"0000")
    g.add_tensor(w)
    n_conv3 = Node("Conv", [v_in, "w"], [v_conv])
    n_bn3 = Node("BatchNormalization", [v_conv, "s", "b"], [v_out])  # missing mean, var
    g.add_node(n_conv3)
    g.add_node(n_bn3)
    fuse_conv_bn(g)
    assert n_bn3 in g.nodes

    # 4. BN with missing tensors in graph
    g = Graph("fuse_invalid4")
    g.add_tensor(v_in)
    g.add_tensor(v_conv)
    g.add_tensor(v_out)
    g.add_tensor(w)
    n_conv4 = Node("Conv", [v_in, "w"], [v_conv])
    n_bn4 = Node("BatchNormalization", [v_conv, "s", "b", "m", "v"], [v_out])
    # Tensors s, b, m, v are NOT in g.tensors
    g.add_node(n_conv4)
    g.add_node(n_bn4)
    fuse_conv_bn(g)
    assert n_bn4 in g.nodes


def test_reconstruct_sequences_multi_output():
    """Test sequence reconstruction with multiple outputs."""
    g = Graph("test")
    n1 = Node("Relu", ["in"], ["out1", "out2"])
    g.add_node(n1)
    reconstruct_sequences(g)


def test_reconstruct_sequences_cycle():
    """Test sequence reconstruction with a cycle."""
    g = Graph("test")
    v1 = Variable("in")
    v2 = Variable("out1")
    g.add_tensor(v1)
    g.add_tensor(v2)
    n1 = Node("Relu", [v1], [v2])
    n2 = Node("Relu", [v2], [v1])
    v2.outputs.append(n2)
    v1.outputs.append(n1)
    g.add_node(n1)
    g.add_node(n2)
    reconstruct_sequences(g)


def test_reconstruct_sequences_primary_inputs_robust():
    """Test sequence reconstruction with primary inputs and strings."""
    g = Graph("test")
    v_in = Variable("in")
    g.add_tensor(v_in)
    # n2 has one primary input (out1) and one external (ext)
    n1 = Node("Relu", ["in"], ["out1"])
    n2 = Node("Add", ["out1", "ext"], ["out2"])
    g.add_node(n1)
    g.add_node(n2)
    # out1 is in graph.consumer_map but not in graph.tensors
    reconstruct_sequences(g)


def test_reconstruct_sequences_primary_inputs_non_initializer():
    """Test sequence reconstruction where input is a non-initializer variable."""
    g = Graph("test")
    v_in = Variable("in")
    v_out1 = Variable("out1")
    v_out2 = Variable("out2")
    g.add_tensor(v_in)
    g.add_tensor(v_out1)
    g.add_tensor(v_out2)

    n1 = Node("Relu", [v_in], [v_out1])
    n2 = Node("Relu", [v_out1], [v_out2])
    g.add_node(n1)
    g.add_node(n2)

    # n2 input 'out1' is a Variable, so it's a primary input.
    # We want to hit the case where it's exactly 1 primary input.
    # And we want to hit the logic where i_t is None or is_initializer is False.
    res = reconstruct_sequences(g)
    assert len(res) == 1


def test_evaluate_math_graph_ops():
    """Test evaluation of various math operations in the graph."""
    g = Graph("test")
    c1 = Constant("c1", values=np.array([10.0], dtype=np.float32).tobytes(), shape=(1,))
    c2 = Constant("c2", values=np.array([2.0], dtype=np.float32).tobytes(), shape=(1,))
    g.add_tensor(c1)
    g.add_tensor(c2)

    # Sub, Mul, Div
    g.add_node(Node("Sub", ["c1", "c2"], ["out_sub"]))
    g.add_node(Node("Sub", ["c1"], ["out_neg"]))  # Unary Sub
    g.add_node(Node("Mul", ["c1", "c2"], ["out_mul"]))
    g.add_node(Node("Div", ["c1", "c2"], ["out_div"]))

    # MatMul
    c3 = Constant("c3", values=np.array([[1, 2], [3, 4]], dtype=np.float32).tobytes(), shape=(2, 2))
    c4 = Constant("c4", values=np.array([[5, 6], [7, 8]], dtype=np.float32).tobytes(), shape=(2, 2))
    g.add_tensor(c3)
    g.add_tensor(c4)
    g.add_node(Node("MatMul", ["c3", "c4"], ["out_matmul"]))

    # Transpose
    g.add_node(
        Node(
            "Transpose",
            ["c3"],
            ["out_trans"],
            attributes={"perm": Attribute("perm", "INTS", [1, 0])},
        )
    )

    # Reshape
    c_shape = Constant(
        "c_shape", values=np.array([4], dtype=np.int64).tobytes(), shape=(1,), dtype=DType.INT64
    )
    g.add_tensor(c_shape)
    g.add_node(Node("Reshape", ["c3", "c_shape"], ["out_reshape"]))

    # Cast
    g.add_node(
        Node("Cast", ["c1"], ["out_cast"], attributes={"to": Attribute("to", "INT", 6)})
    )  # Cast to INT32

    res = evaluate_math_graph(g)
    assert res is not None


def test_evaluate_math_graph_missing_input():
    """Test evaluate_math_graph when an input is missing from env."""
    g = Graph("test")
    c1 = Constant("c1", values=b"0000")
    g.add_tensor(c1)
    n = Node("Add", ["c1", "missing"], ["out"])
    g.add_node(n)
    res = evaluate_math_graph(g)
    assert res is None


def test_numpy_import_error_coverage():
    """Test ImportError branches for numpy-dependent functions."""
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Mock import."""
        if name == "numpy":
            raise ImportError("mock")
        return real_import(name, *args, **kwargs)

    c = Constant("c", values=bytearray(4), shape=(1,), dtype=DType.FLOAT32)

    with patch("builtins.__import__", side_effect=mock_import):
        # transpose_constant
        res_t = transpose_constant(c, [0])
        assert res_t == c

        # evaluate_math_graph (it uses np = None)
        g = Graph("test")
        c1 = Constant("c1", values=struct.pack("<f", 10.0), shape=(), dtype=DType.FLOAT32)
        c2 = Constant("c2", values=struct.pack("<f", 2.0), shape=(), dtype=DType.FLOAT32)
        g.add_tensor(c1)
        g.add_tensor(c2)
        g.add_node(Node("Add", ["c1", "c2"], ["out"]))
        res_eval = evaluate_math_graph(g)
        assert res_eval is not None
        assert struct.unpack("<f", res_eval.data)[0] == 12.0


def test_extract_scalar_coverage():
    """Test extract_scalar coverage."""
    assert extract_scalar(Constant("none", values=None)) is None
    assert extract_scalar(Constant("bad", values=b"12", dtype=DType.FLOAT32)) is None


def test_merge_lora_adapters_noop():
    """Test merging LoRA adapters when no changes are needed."""
    g = Graph("test")
    c_lora_a = Constant("model.layers.0.lora_a", values=bytearray(4))
    c_lora_b = Constant("model.layers.0.lora_b", values=bytearray(4))
    g.add_tensor(c_lora_a)
    g.add_tensor(c_lora_b)
    merge_lora_adapters(g)


def test_evaluate_math_graph_no_nodes():
    """Test evaluation of a math graph with no nodes."""
    assert evaluate_math_graph(Graph("empty")) is None
