"""Unit tests for symbolic ONNX script Variable operator overloading."""

import pytest
from onnx9000.toolkit.script.var import Var


def test_script_var_ops() -> None:
    """Verify arithmetic, relational, and indexing operators on Var return new Var instances."""
    v = Var("x")

    assert isinstance(v + 1, Var)
    assert isinstance(1 + v, Var)
    assert isinstance(v - 1, Var)
    assert isinstance(1 - v, Var)
    assert isinstance(v * 1, Var)
    assert isinstance(1 * v, Var)
    assert isinstance(v / 1, Var)
    assert isinstance(1 / v, Var)
    assert isinstance(v**1, Var)
    assert isinstance(v @ 1, Var)
    assert isinstance(v < 1, Var)
    assert isinstance(v > 1, Var)
    assert isinstance(v == 1, Var)
    assert isinstance(v != 1, Var)
    assert isinstance(v[1], Var)

    with pytest.raises(TypeError):
        _ = v % 1
    with pytest.raises(TypeError):
        _ = v <= 1
    with pytest.raises(TypeError):
        _ = v >= 1
    with pytest.raises(TypeError):
        v(1)  # type: ignore[operator]
    with pytest.raises(AttributeError):
        _ = v.test_attr


def test_script_var_ops_more() -> None:
    """Verify bitwise operators and slice indexing on Var return new Var instances."""
    v = Var("x")

    assert isinstance(v - 1, Var)
    assert isinstance(1 - v, Var)
    assert isinstance(v & 1, Var)
    assert isinstance(v | 1, Var)
    assert isinstance(v ^ 1, Var)
    assert isinstance(~v, Var)
    assert isinstance(v[1:5:2], Var)


def test_script_var_rename() -> None:
    """Verify Var.rename updates variable name in-place and returns self."""
    v = Var("x")
    renamed = v.rename("y")
    assert renamed is v
    assert repr(v) == "Var(y)"
