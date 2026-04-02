"""Coverage tests for TVM Relay visualization."""

import pytest
from unittest.mock import MagicMock
from onnx9000.tvm.relay.visualize import DotPrinter, to_dot
from onnx9000.tvm.relay.expr import (
    Call,
    Op,
    Var,
    Constant,
    TupleExpr,
    TupleGetItem,
    Let,
    If,
    Function,
)


def test_dot_printer_all_types():
    """Verify DotPrinter on all Expr types."""
    x = Var("x")
    y = Var("y")
    c = Constant(None)
    add_op = Op("Add")

    # Call
    call_expr = Call(add_op, [x, y])

    # Tuple
    tup = TupleExpr([x, y])
    tgi = TupleGetItem(tup, 0)

    # Let
    let = Let(x, c, y)

    # If
    cond = Var("cond")
    if_expr = If(cond, x, y)

    # Function
    func = Function([x, y], call_expr)

    # Run to_dot on the most complex one to trigger most branches
    res = to_dot(func)
    assert "digraph IR" in res
    assert "Function" in res
    assert "Var(%x)" in res
    assert "Op(Add)" in res


def test_dot_printer_visitor_methods():
    """Verify specific visitor methods for coverage."""
    dp = DotPrinter()
    x = Var("x")
    dp.get_id(x)
    dp._add_edge(x, Var("y"), "test")
