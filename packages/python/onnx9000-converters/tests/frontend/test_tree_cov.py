"""Tests for frontend tree."""

from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.converters.frontend.tree import find_tensors, tree_flatten, tree_map, tree_unflatten
from onnx9000.core.dtypes import DType


def test_tree_map():
    """Docstring for D103."""

    def inc(x):
        """Inc."""
        return x + 1

    assert tree_map(inc, 1) == 2
    assert tree_map(inc, [1, 2]) == [2, 3]
    assert tree_map(inc, (1, 2)) == (2, 3)
    assert tree_map(inc, {"a": 1, "b": 2}) == {"a": 2, "b": 3}
    assert tree_map(inc, [{"a": 1}, (2,)]) == [{"a": 2}, (3,)]


def test_tree_flatten_unflatten():
    """Docstring for D103."""
    tree = [{"a": 1}, (2, 3), 4]
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2, 3, 4]
    reconstructed = tree_unflatten(leaves, treedef)
    assert reconstructed == tree

    # Extra coverage for empty structures and complex nesting
    tree2 = {"b": [5, (6,)], "c": {}}
    leaves2, treedef2 = tree_flatten(tree2)
    assert leaves2 == [5, 6]
    assert tree_unflatten(leaves2, treedef2) == tree2


def test_find_tensors():
    """Docstring for D103."""
    t1 = Tensor(name="t1", shape=(1,), dtype=DType.FLOAT32)
    t2 = Tensor(name="t2", shape=(2,), dtype=DType.FLOAT32)

    tree = [{"a": t1}, (t2,), "not_tensor"]
    tensors = find_tensors(tree)
    assert len(tensors) == 2
    assert any(t.name == "t1" for t in tensors)
    assert any(t.name == "t2" for t in tensors)
