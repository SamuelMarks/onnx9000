"""Tests for parsing if-statements with multiple outputs."""

from onnx9000.toolkit.script import op, script


def my_if(cond, a, b):
    """A sample function containing an if-statement with two outputs."""
    if cond:
        x = op.Add(a, b)
        y = op.Sub(a, b)
    else:
        x = op.Mul(a, b)
        y = op.Div(a, b)
    return x, y


def test_if_multiple_outputs():
    """Test that a script with a two-output if-statement is parsed correctly."""
    s = script(my_if)
    builder = s.to_builder()
    assert len(builder.outputs) == 2


def my_if_2(cond, a, b):
    """A sample function containing an if-statement with three outputs."""
    if cond:
        x = op.Add(a, b)
        y = op.Sub(a, b)
        z = op.Add(x, y)
    else:
        x = op.Mul(a, b)
        y = op.Div(a, b)
        z = op.Sub(x, y)
    return x, y, z


def test_if_more_outputs():
    """Test that a script with a three-output if-statement is parsed correctly."""
    s = script(my_if_2)
    builder = s.to_builder()
    assert len(builder.outputs) == 3
