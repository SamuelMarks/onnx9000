"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Node
from onnx9000.training.autograd.rules import get_vjp_rule


def test_reshape_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Reshape", ["a", "shape"], ["c"], {}, name="reshape_node")
    rule = get_vjp_rule("Reshape")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_transpose_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Transpose", ["a"], ["c"], {"perm": [1, 0]}, name="transpose_node")
    rule = get_vjp_rule("Transpose")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    node2 = Node("Transpose", ["a"], ["c"], {}, name="transpose_node2")
    nodes2, names2 = rule.build_backward_nodes(node2, ["grad_c"])
    assert len(nodes2) == 1


def test_squeeze_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Squeeze", ["a", "axes"], ["c"], {}, name="squeeze_node")
    rule = get_vjp_rule("Squeeze")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_unsqueeze_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Unsqueeze", ["a", "axes"], ["c"], {}, name="unsqueeze_node")
    rule = get_vjp_rule("Unsqueeze")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_flatten_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Flatten", ["a"], ["c"], {}, name="flatten_node")
    rule = get_vjp_rule("Flatten")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_concat_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Concat", ["a", "b"], ["c"], {"axis": 0}, name="concat_node")
    rule = get_vjp_rule("Concat")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert len(names) == 2


def test_split_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Split", ["a"], ["b", "c"], {"axis": 0}, name="split_node")
    rule = get_vjp_rule("Split")
    nodes, names = rule.build_backward_nodes(node, ["grad_b", "grad_c"])
    assert len(nodes) == 1
    assert len(names) == 1


def test_slice_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Slice", ["a", "starts", "ends", "axes"], ["c"], {}, name="slice_node")
    rule = get_vjp_rule("Slice")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_gather_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Gather", ["a", "indices"], ["c"], {}, name="gather_node")
    rule = get_vjp_rule("Gather")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_gatherelements_vjp():
    """Provides semantic functionality and verification."""
    node = Node(
        "GatherElements", ["a", "indices"], ["c"], {}, name="gatherelements_node"
    )
    rule = get_vjp_rule("GatherElements")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_gathernd_vjp():
    """Provides semantic functionality and verification."""
    node = Node("GatherND", ["a", "indices"], ["c"], {}, name="gathernd_node")
    rule = get_vjp_rule("GatherND")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_scatter_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Scatter", ["a", "indices", "updates"], ["c"], {}, name="scatter_node")
    rule = get_vjp_rule("Scatter")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_scatternd_vjp():
    """Provides semantic functionality and verification."""
    node = Node(
        "ScatterND", ["a", "indices", "updates"], ["c"], {}, name="scatternd_node"
    )
    rule = get_vjp_rule("ScatterND")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_scatterelements_vjp():
    """Provides semantic functionality and verification."""
    node = Node(
        "ScatterElements",
        ["a", "indices", "updates"],
        ["c"],
        {},
        name="scatterelements_node",
    )
    rule = get_vjp_rule("ScatterElements")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_tile_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Tile", ["a", "repeats"], ["c"], {}, name="tile_node")
    rule = get_vjp_rule("Tile")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_pad_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Pad", ["a", "pads"], ["c"], {}, name="pad_node")
    rule = get_vjp_rule("Pad")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_cast_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Cast", ["a"], ["c"], {"to": 1}, name="cast_node")
    rule = get_vjp_rule("Cast")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_expand_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Expand", ["a", "shape"], ["c"], {}, name="expand_node")
    rule = get_vjp_rule("Expand")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_where_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Where", ["cond", "x", "y"], ["c"], {}, name="where_node")
    rule = get_vjp_rule("Where")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert len(names) == 2


def test_nonzero_vjp():
    """Provides semantic functionality and verification."""
    node = Node("NonZero", ["a"], ["c"], {}, name="nonzero_node")
    rule = get_vjp_rule("NonZero")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
