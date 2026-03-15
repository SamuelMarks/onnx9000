"""Test Containers."""

from collections import OrderedDict
from onnx9000.frontends.frontend.nn.containers import (
    Sequential,
    ModuleList,
    ModuleDict,
    ParameterList,
)
from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.frontends.frontend.tensor import Parameter, Tensor
from onnx9000.core.dtypes import DType
import pytest


class Dummy(Module):
    """Provides semantic functionality and verification."""

    def __init__(self, c=1):
        """Provides semantic functionality and verification."""
        super().__init__()
        self.c = c

    def forward(self, x):
        """Provides semantic functionality and verification."""
        return x + self.c


def test_sequential():
    """Provides semantic functionality and verification."""
    seq1 = Sequential(Dummy(1), Dummy(2))
    assert len(list(seq1.children())) == 2
    seq2 = Sequential(OrderedDict([("one", Dummy(1)), ("two", Dummy(2))]))
    assert len(list(seq2.children())) == 2
    assert "one" in seq2._modules
    from onnx9000 import GraphBuilder, Tracing

    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((1,), DType.FLOAT32, "x")
        y = seq1(x)
    assert y is not None


def test_module_list():
    """Provides semantic functionality and verification."""
    ml = ModuleList([Dummy(1), Dummy(2)])
    ml.append(Dummy(3))
    assert len(ml) == 3
    assert len(list(ml)) == 3
    ml.extend([Dummy(4), Dummy(5)])
    assert len(ml) == 5
    assert isinstance(ml[0], Dummy)
    assert isinstance(ml[-1], Dummy)
    ml_slice = ml[1:3]
    assert isinstance(ml_slice, ModuleList)
    assert len(ml_slice) == 2


def test_module_dict():
    """Provides semantic functionality and verification."""
    md = ModuleDict({"a": Dummy(1)})
    md.update({"b": Dummy(2)})
    md.update([("c", Dummy(3))])
    md["d"] = Dummy(4)
    assert len(list(md.keys())) == 4
    assert len(list(md.values())) == 4
    assert len(list(md.items())) == 4
    assert isinstance(md["a"], Dummy)
    with pytest.raises(TypeError):
        md.update(1)


def test_parameter_list():
    """Provides semantic functionality and verification."""
    pl = ParameterList([Parameter((1,), DType.FLOAT32)])
    pl.append(Parameter((2,), DType.FLOAT32))
    pl.extend([Parameter((3,), DType.FLOAT32)])
    assert len(pl) == 3
    assert pl[0].shape == (1,)
    assert pl[-1].shape == (3,)
    assert len(list(pl)) == 3
