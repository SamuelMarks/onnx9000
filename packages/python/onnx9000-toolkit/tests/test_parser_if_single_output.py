"""Tests for parsing if-statements with a single output."""

import pytest
from onnx9000.toolkit.script import op, script


def my_if_single(cond, a, b):
    """A sample function containing an if-statement with one output."""
    if cond:
        x = op.Add(a, b)
    else:
        x = op.Mul(a, b)
    return x


def test_if_single_output():
    """Test that a script with a single-output if-statement is parsed correctly."""
    s = script(my_if_single)
    builder = s.to_builder()
    assert len(builder.outputs) == 1
