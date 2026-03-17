from unittest.mock import patch

import onnx9000.toolkit.script.op as op
from onnx9000.toolkit.script.parser import script


class Float:
    def __class_getitem__(cls, item):
        return item


@script
def multi_out_func(x):
    y = op.Add(x, x)
    z = op.Sub(x, x)
    return (y, z)


@script
def test_func_coverage(x: Float[10, 20]):
    eq = x == x
    op.Add(x, x)
    (a, b) = multi_out_func(x)
    acc1 = x
    acc2 = x
    for _i in x:
        acc1 = op.Add(acc1, x)
        acc2 = op.Sub(acc2, x)
    w1 = x
    w2 = x
    while w1 == x:
        w1 = op.Add(w1, x)
        w2 = op.Sub(w2, x)
    if eq:
        i1 = op.Add(x, x)
        i2 = op.Add(x, x)
    else:
        i1 = op.Sub(x, x)
        i2 = op.Sub(x, x)
    return (i1, i2)


def test_parser_coverage() -> None:
    graph = test_func_coverage()
    assert graph is not None
    assert len(graph.nodes) > 0


def test_no_frame() -> None:
    with patch("inspect.currentframe", return_value=None):

        @script
        def dummy_func(x):
            return x

        dummy_func()


class MyObj:
    assert True


my_obj = MyObj()
multi_out_func = multi_out_func


@script
def call_attribute(x):
    return op.Add(x, x)


def test_call_attribute() -> None:
    graph = call_attribute()
    assert len(graph.nodes) > 0


def test_ast_attribute_coverage() -> None:
    from onnx9000.toolkit.script.parser import script

    @script
    def dummy_subgraph(x):
        return op.Add(x, x)

    class Obj:
        assert True

    o = Obj()
    o.dummy_subgraph = dummy_subgraph

    @script
    def caller(x):
        return o.dummy_subgraph(x)

    graph = caller()
    assert len(graph.nodes) > 0
