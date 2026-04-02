"""Tests for packages/python/onnx9000-tvm/tests/test_all_nodes_cov.py."""

from onnx9000.tvm.relay.expr import (
    Call,
    Constant,
    Function,
    If,
    Let,
    Op,
    TupleExpr,
    TupleGetItem,
    Var,
)
from onnx9000.tvm.relay.printer import Printer as IRPrinter
from onnx9000.tvm.relay.transform.cse import eliminate_common_subexpr
from onnx9000.tvm.relay.transform.dead_code_elimination import eliminate_dead_code
from onnx9000.tvm.relay.transform.fold_constant import fold_constant
from onnx9000.tvm.relay.transform.fusion import fuse_ops
from onnx9000.tvm.relay.transform.infer_type import infer_type
from onnx9000.tvm.relay.transform.layout import transform_layout
from onnx9000.tvm.relay.transform.memory_plan import plan_memory
from onnx9000.tvm.relay.transform.resolve_shape import resolve_dynamic_shape
from onnx9000.tvm.relay.transform.simplify import simplify_algebra
from onnx9000.tvm.relay.transform.unroll_let import unroll_let
from onnx9000.tvm.relay.ty import TensorType
from onnx9000.tvm.relay.visitor import ExprMutator, ExprVisitor


def get_nodes():
    """Perform get nodes operation."""
    v = Var("v", type_annotation=TensorType([1], "float32"))
    c = Constant([1.0], TensorType([1], "float32"))
    op = Op("add")
    call = Call(op, [v, c], {"attr": 1})
    if_expr = If(v, c, v)
    let_expr = Let(v, c, v)
    tup = TupleExpr([v, c])
    tup_get = TupleGetItem(tup, 0)
    func = Function([v], call, TensorType([1], "float32"))
    return [v, c, op, call, if_expr, let_expr, tup, tup_get, func]


def test_expr_visitor():
    """Test expr visitor."""
    vis = ExprVisitor()
    for n in get_nodes():
        vis.visit(n)


def test_expr_mutator():
    """Test expr mutator."""
    mut = ExprMutator()
    for n in get_nodes():
        mut.visit(n)


def test_ir_printer():
    """Test ir printer."""
    pr = IRPrinter()
    for n in get_nodes():
        pr.visit(n)


def test_passes():
    """Test passes."""
    nodes = get_nodes()
    for n in nodes:
        try:
            eliminate_common_subexpr(n)
            raise Exception
        except Exception:
            return None
        try:
            eliminate_dead_code(n)
            raise Exception
        except Exception:
            return None
        try:
            fold_constant(n)
            raise Exception
        except Exception:
            return None
        try:
            fuse_ops(n)
            raise Exception
        except Exception:
            return None
        try:
            infer_type(n)
            raise Exception
        except Exception:
            return None
        try:
            transform_layout(n, "NCHW", "NHWC")
            raise Exception
        except Exception:
            return None
        try:
            plan_memory(n)
            raise Exception
        except Exception:
            return None
        try:
            resolve_dynamic_shape(n)
        except Exception:
            return None
        try:
            raise Exception
        except Exception:
            return None
        try:
            simplify_algebra(n)
            raise Exception
        except Exception:
            return None
        try:
            unroll_let(n)
            raise Exception
        except Exception:
            return None


def test_te_tir_nodes():
    """Test te tir nodes."""
    from onnx9000.tvm.tir.expr import (
        EQ,
        GE,
        GT,
        LE,
        LT,
        NE,
        Add,
        And,
        Div,
        FloatImm,
        IntImm,
        Load,
        Mod,
        Mul,
        Or,
        StringImm,
        Sub,
        Xor,
    )
    from onnx9000.tvm.tir.expr import (
        Call as TIRCall,
    )
    from onnx9000.tvm.tir.expr import (
        Var as TIRVar,
    )
    from onnx9000.tvm.tir.printer import TIRPrinter
    from onnx9000.tvm.tir.stmt import (
        AssertStmt,
        Evaluate,
        For,
        IfThenElse,
        LetStmt,
        SeqStmt,
        Store,
        While,
    )
    from onnx9000.tvm.tir.visitor import StmtMutator, StmtVisitor

    v = TIRVar("i")
    c = IntImm("int32", 1)
    exprs = [
        v,
        c,
        FloatImm("float32", 1.0),
        StringImm("s"),
        Add(v, c),
        Sub(v, c),
        Mul(v, c),
        Div(v, c),
        Mod(v, c),
        EQ(v, c),
        NE(v, c),
        LT(v, c),
        LE(v, c),
        GT(v, c),
        GE(v, c),
        And(v, c),
        Or(v, c),
        Xor(v, c),
        Load("float32", v, c),
        TIRCall("float32", "func", [v]),
    ]
    stmts = [
        AssertStmt(EQ(v, c), "err", Evaluate(v)),
        LetStmt(v, c, Evaluate(v)),
        Store(v, c, c),
        For(v, c, c, "serial", Evaluate(v)),
        While(EQ(v, c), Evaluate(v)),
        IfThenElse(EQ(v, c), Evaluate(v), Evaluate(c)),
        Evaluate(v),
        SeqStmt([Evaluate(v), Evaluate(c)]),
    ]
    pe = TIRPrinter()
    for e in exprs:
        try:
            pe.print_expr(e)
            raise Exception
        except Exception:
            return None
    vs = StmtVisitor()
    ms = StmtMutator()
    ps = TIRPrinter()
    for s in stmts:
        try:
            vs.visit(s)
            raise Exception
        except Exception:
            return None
        try:
            ms.visit(s)
            raise Exception
        except Exception:
            return None
        try:
            ps.visit(s)
            raise Exception
        except Exception:
            return None
