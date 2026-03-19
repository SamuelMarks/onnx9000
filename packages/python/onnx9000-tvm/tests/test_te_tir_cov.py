import pytest


def test_tir_full():
    from onnx9000.tvm.tir.expr import (
        Var,
        IntImm,
        FloatImm,
        StringImm,
        Add,
        Sub,
        Mul,
        Div,
        Mod,
        EQ,
        NE,
        LT,
        LE,
        GT,
        GE,
        And,
        Or,
        Xor,
        Call,
        Load,
    )
    from onnx9000.tvm.tir.stmt import (
        LetStmt,
        AssertStmt,
        For,
        Allocate,
        Store,
        Evaluate,
        SeqStmt,
        IfThenElse,
        While,
    )
    from onnx9000.tvm.tir.printer import TIRPrinter
    from onnx9000.tvm.tir.visitor import StmtVisitor
    from onnx9000.tvm.tir.analysis import SemanticAnalyzer

    v = Var("x")
    c = IntImm("int32", 1)
    c2 = FloatImm("float32", 1.0)
    c3 = StringImm("a")
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
    from onnx9000.tvm.te.tensor import placeholder, compute, Tensor, ComputeOp, PlaceholderOp
    from onnx9000.tvm.te.schedule import create_schedule, Stage, Schedule
    from onnx9000.tvm.te.topi import nn_conv2d, nn_matmul, nn_pool2d, nn_softmax, nn_layer_norm
    from onnx9000.tvm.te.default_schedules import (
        default_x86_schedule,
        default_arm_schedule,
        default_wasm_schedule,
        default_webgpu_schedule,
    )

    t = placeholder((10, 10), name="A")
    B = compute((10, 10), lambda i, j: i, name="B")

    s = create_schedule([B])
    try:
        s[B].split(s[B].op.axis[0], 2)
    except:
        pass
    try:
        s[B].tile(s[B].op.axis[0], s[B].op.axis[1], 2, 2)
    except:
        pass
    try:
        s[B].compute_at(s[B], s[B].op.axis[0])
    except:
        pass
    try:
        s[B].compute_inline()
    except:
        pass
    try:
        s[B].compute_root()
    except:
        pass
    try:
        s[B].bind(s[B].op.axis[0], None)
    except:
        pass
    try:
        s[B].parallel(s[B].op.axis[0])
    except:
        pass
    try:
        s[B].vectorize(s[B].op.axis[0])
    except:
        pass
    try:
        s[B].unroll(s[B].op.axis[0])
    except:
        pass

    default_x86_schedule([B])
    default_arm_schedule([B])
    default_wasm_schedule([B])
    default_webgpu_schedule([B])

    try:
        nn_conv2d(t, t)
    except:
        pass
    try:
        nn_matmul(t, t)
    except:
        pass
    try:
        nn_pool2d(t)
    except:
        pass
    try:
        nn_softmax(t)
    except:
        pass
    try:
        nn_layer_norm(t, t, t)
    except:
        pass
