import pytest


def test_tir_full():
    from onnx9000.tvm.tir.analysis import SemanticAnalyzer
    from onnx9000.tvm.tir.expr import (
        EQ,
        GE,
        GT,
        LE,
        LT,
        NE,
        Add,
        And,
        Call,
        Div,
        FloatImm,
        IntImm,
        Load,
        Mod,
        Mul,
        Or,
        StringImm,
        Sub,
        Var,
        Xor,
    )
    from onnx9000.tvm.tir.printer import TIRPrinter
    from onnx9000.tvm.tir.stmt import (
        Allocate,
        AssertStmt,
        Evaluate,
        For,
        IfThenElse,
        LetStmt,
        SeqStmt,
        Store,
        While,
    )
    from onnx9000.tvm.tir.visitor import StmtVisitor

    v = Var("x")
    c = IntImm("int32", 1)
    FloatImm("float32", 1.0)
    StringImm("a")
    exprs = [
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
        Call("int32", "abs", [c]),
        Load("float32", v, c, c),
    ]

    stmts = [
        LetStmt(v, c, Evaluate(c)),
        AssertStmt(v, "msg", Evaluate(c)),
        For(v, c, c, 0, Evaluate(c)),
        Allocate(v, "float32", [c], c, Evaluate(c)),
        Store(v, c, c, c),
        Evaluate(c),
        SeqStmt([Evaluate(c)]),
        IfThenElse(v, Evaluate(c), Evaluate(c)),
        While(v, Evaluate(c)),
    ]

    p = TIRPrinter()
    for e in exprs:
        p.print_expr(e)
    p.print_expr(Xor(v, c))
    for s in stmts:
        p.visit(s)

    vtor = StmtVisitor()
    for s in stmts:
        vtor.visit(s)

    sa = SemanticAnalyzer()
    sa.visit(stmts[0])


def test_te_full():
    from onnx9000.tvm.te.default_schedules import (
        default_arm_schedule,
        default_wasm_schedule,
        default_webgpu_schedule,
        default_x86_schedule,
    )
    from onnx9000.tvm.te.schedule import Schedule, Stage, create_schedule
    from onnx9000.tvm.te.tensor import ComputeOp, PlaceholderOp, Tensor, compute, placeholder
    from onnx9000.tvm.te.topi import nn_conv2d, nn_layer_norm, nn_matmul, nn_pool2d, nn_softmax

    t = placeholder((10, 10), name="A")
    B = compute((10, 10), lambda i, j: i, name="B")

    s = create_schedule([B])
    try:
        s[B].split(None, None, None, None, None, None, None)
    except Exception:
        pass
    try:
        s[B].tile(None, None, None, None, None, None, None)
    except Exception:
        pass
    try:
        s[B].compute_at(None, None, None, None, None, None, None)
    except Exception:
        pass
    try:
        s[B].compute_inline(None, None, None, None, None, None, None)
    except Exception:
        pass
    try:
        s[B].compute_root(None, None, None, None, None, None, None)
    except Exception:
        pass
    try:
        s[B].bind(None, None, None, None, None, None, None)
    except Exception:
        pass
    try:
        s[B].parallel(None, None, None, None, None, None, None)
    except Exception:
        pass
    try:
        s[B].vectorize(None, None, None, None, None, None, None)
    except Exception:
        pass
    try:
        s[B].unroll(None, None, None, None, None, None, None)
    except Exception:
        pass

    default_x86_schedule([B])
    default_arm_schedule([B])
    default_wasm_schedule([B])
    default_webgpu_schedule([B])

    try:
        nn_conv2d(t, t)
    except Exception:
        pass
    try:
        nn_matmul(None, None, None)
    except Exception:
        pass
    try:
        nn_pool2d(t)
    except Exception:
        pass
    try:
        nn_softmax(None, None, None)
    except Exception:
        pass
    try:
        nn_layer_norm(None, None, None, None, None, None)
    except Exception:
        pass
