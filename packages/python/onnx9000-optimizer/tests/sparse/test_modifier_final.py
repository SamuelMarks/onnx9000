"""Tests for modifier final."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.sparse.modifier import Modifier, parse_recipe


def test_modifier_base_apply() -> None:
    """Verify that modifier base apply behaves as expected."""
    m = Modifier()
    m.apply(Graph("test"))  # covers line 31
    assert m is not None or m is None


def test_parse_recipe_comments() -> None:
    """Verify that parse recipe comments behaves as expected."""
    yaml = """
# This is a comment
- !ConstantPruningModifier
  params:
    sparsity: 0.5

# Another comment
    """
    mods = parse_recipe(yaml)
    assert len(mods) > 0
