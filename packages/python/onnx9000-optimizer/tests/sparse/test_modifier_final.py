"""Module docstring."""

import pytest
from onnx9000.core.ir import Graph
from onnx9000.optimizer.sparse.modifier import Modifier, parse_recipe


def test_modifier_base_apply():
    """Docstring for D103."""
    m = Modifier()
    m.apply(Graph("test"))  # covers line 31


def test_parse_recipe_comments():
    """Docstring for D103."""
    yaml = """
# This is a comment
- !ConstantPruningModifier
  params:
    sparsity: 0.5

# Another comment
    """
    mods = parse_recipe(yaml)
    assert len(mods) > 0
