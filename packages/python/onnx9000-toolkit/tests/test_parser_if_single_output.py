"""Tests for parsing if-statements with a single output."""

from onnx9000.toolkit.script import op, script


def my_if_single(
    cond: int | float | str | bool | bytes | list | dict | tuple | None,
    a: int | float | str | bool | bytes | list | dict | tuple | None,
    b: int | float | str | bool | bytes | list | dict | tuple | None,
) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """A sample function containing an if-statement with one output."""
    x = op.Add(a, b) if cond else op.Mul(a, b)
    return x


def test_if_single_output() -> None:
    """Test that a script with a single-output if-statement is parsed correctly."""
    s = script(my_if_single)
    builder = s.to_builder()
    assert len(builder.outputs) == 1
