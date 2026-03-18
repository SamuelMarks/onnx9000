import pytest
from onnx9000.toolkit.script import script
from onnx9000.toolkit.script import op


def my_if_single(cond, a, b):
    if cond:
        x = op.Add(a, b)
    else:
        x = op.Mul(a, b)
    return x


def test_if_single_output():
    s = script(my_if_single)
    builder = s.to_builder()
    assert len(builder.outputs) == 1
