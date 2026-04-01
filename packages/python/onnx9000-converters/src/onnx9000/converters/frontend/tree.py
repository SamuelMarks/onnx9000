"""Tree utility for handling nested collections (list, tuple, dict)."""

from typing import Any, Callable, List, Tuple


def tree_map(fn: Callable, tree: Any) -> Any:
    """Recursively apply a function to all leaves in a tree."""
    if isinstance(tree, list):
        return [tree_map(fn, x) for x in tree]
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, x) for x in tree)
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    return fn(tree)


def tree_flatten(tree: Any) -> Tuple[List[Any], Any]:
    """Flatten a tree into a list of leaves and a treedef."""
    leaves = []

    def _flatten(t):
        if isinstance(t, list):
            return [_flatten(x) for x in t], list
        if isinstance(t, tuple):
            return [_flatten(x) for x in t], tuple
        if isinstance(t, dict):
            return {k: _flatten(v) for k, v in t.items()}, dict
        leaves.append(t)
        return len(leaves) - 1, None

    treedef = _flatten(tree)
    return leaves, treedef


def tree_unflatten(leaves: List[Any], treedef: Any) -> Any:
    """Reconstruct a tree from leaves and a treedef."""
    structure, container = treedef
    if container is list:
        return [tree_unflatten(leaves, x) for x in structure]
    if container is tuple:
        return tuple(tree_unflatten(leaves, x) for x in structure)
    if container is dict:
        return {k: tree_unflatten(leaves, v) for k, v in structure.items()}
    return leaves[structure]


def find_tensors(tree: Any) -> List[Any]:
    """Recursively find all Tensor objects in a tree."""
    from onnx9000.converters.frontend.tensor import Tensor

    tensors = []

    def _find(t):
        if isinstance(t, Tensor):
            tensors.append(t)
        elif isinstance(t, (list, tuple)):
            for x in t:
                _find(x)
        elif isinstance(t, dict):
            for v in t.values():
                _find(v)

    _find(tree)
    return tensors
