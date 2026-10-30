"""Tests to cover remaining gaps in core module."""

import struct

import pytest
from onnx9000.core.ir import (
    Attribute,
    Constant,
    DType,
    DynamicDim,
    Graph,
    Node,
    Variable,
)
from onnx9000.core.registry import UnsupportedOpError, global_registry, register_op
from onnx9000.core.shape_inference import get_attr, infer_shapes_and_types
from onnx9000.core.surgeon import cleanup, fold_constants_math, reconstruct_sequences
from onnx9000.core.symbolic import simplify_expression


def test_reconstruct_sequences() -> None:
    """Test reconstructing sequences from linear chains of nodes."""
    g = Graph("test")
    v1 = Variable("v1", [1, 10], DType.FLOAT32)
    v2 = Variable("v2", [1, 10], DType.FLOAT32)
    v3 = Variable("v3", [1, 10], DType.FLOAT32)
    v4 = Variable("v4", [1, 10], DType.FLOAT32)
    g.add_tensor(v1)
    g.add_tensor(v2)
    g.add_tensor(v3)
    g.add_tensor(v4)

    n1 = Node("Relu", ["v1"], ["v2"])
    n2 = Node("Relu", ["v2"], ["v3"])
    n3 = Node("Relu", ["v3"], ["v4"])
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)

    seqs = reconstruct_sequences(g)
    assert len(seqs) == 1
    assert len(next(iter(seqs.values()))) == 3


def test_shape_inference_gaps() -> None:
    """Test shape inference for ops with coverage gaps."""
    g = Graph("test")
    v1 = Variable("v1", [2, 3, 4], DType.FLOAT32)
    g.add_tensor(v1)

    # Flatten
    n_flat = Node(
        "Flatten",
        ["v1"],
        ["out_flat"],
        attributes={"axis": Attribute("axis", "INT", 1)},
    )
    g.add_node(n_flat)
    infer_shapes_and_types(g)
    assert list(g.tensors["out_flat"].shape) == [2, 12]

    # Custom fallback for unknown op
    n_unk = Node("UnknownOp", ["v1"], ["out_unk"])
    g.add_node(n_unk)
    infer_shapes_and_types(g)
    assert isinstance(g.tensors["out_unk"].shape[0], DynamicDim)

    # get_attr fallback
    n_attr = Node("Test", [], [], attributes={"a": "raw_val"})
    assert get_attr(n_attr, "a") == "raw_val"

    # Test get_attr with Attribute object that has name but not in keys
    n_attr2 = Node(
        "Test", [], [], attributes={"none": Attribute("real_name", "STRING", "real_val")}
    )
    assert get_attr(n_attr2, "real_name") == "real_val"


def test_surgeon_fold_math_gaps() -> None:
    """Test folding math ops with different number of inputs."""
    g = Graph("test")
    c1 = Constant("c1", values=struct.pack("<f", 1.0))
    c2 = Constant("c2", values=struct.pack("<f", 2.0))
    g.add_tensor(c1)
    g.add_tensor(c2)

    n_add = Node("Add", ["c1", "c2"], ["out_add"])
    g.add_node(n_add)

    fold_constants_math(g)
    assert "out_add" in g.tensors
    assert isinstance(g.tensors["out_add"], Constant)
    assert g.tensors["out_add"].is_initializer


def test_cleanup_gaps() -> None:
    """Test cleanup with various edge cases."""
    g = Graph("test")
    v_in = Variable("in")
    g.add_tensor(v_in)
    n = Node("Relu", ["in"], [])
    g.add_node(n)
    cleanup(g)
    assert n not in g.nodes


def test_registry_gaps() -> None:
    """Test registry with domain and errors."""
    with pytest.raises(UnsupportedOpError):
        global_registry.get_op("test_domain", "NonExistentOp")

    @register_op("my_domain", "DomainOp")
    def my_op() -> str:
        """My op."""
        return "domain_op_result"

    all_ops = global_registry.get_all_registered()
    assert "my_domain.DomainOp" in all_ops
    assert my_op() == "domain_op_result"


def test_symbolic_simplify() -> None:
    """Test symbolic expression simplification."""
    assert simplify_expression("(x * y) / y") == "x"
    assert simplify_expression("(y * x) / y") == "x"
    assert simplify_expression("x + 0") == "x"
    assert simplify_expression("0 + x") == "x"
    assert simplify_expression("x * 1") == "x"
    assert simplify_expression("1 * x") == "x"
    assert simplify_expression("x * 0") == "0"
    assert simplify_expression("0 * x") == "0"
    assert simplify_expression("x ** 1") == "(x ** 1)"  # Not simplified in current impl
    assert simplify_expression("invalid %$#") == "invalid %$#"
    assert simplify_expression("x // y") == "(x // y)"
    assert simplify_expression("x % y") == "(x % y)"
