import pytest
from onnx9000.toolkit.script import op, script


def my_if(cond, a, b):
    if cond:
        x = op.Add(a, b)
        y = op.Sub(a, b)
    else:
        x = op.Mul(a, b)
        y = op.Div(a, b)
    return x, y


def test_if_multiple_outputs():
    s = script(my_if)
    builder = s.to_builder()
    assert len(builder.outputs) == 2


def my_if_2(cond, a, b):
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
    s = script(my_if_2)
    builder = s.to_builder()
    assert len(builder.outputs) == 3
