def test_wasm():
    from onnx9000.tvm.target.wasm.emitter import (
        generate_ts_typings,
        generate_js_wrapper,
        WASMEmitter,
    )

    generate_ts_typings()
    generate_js_wrapper()
    WASMEmitter().emit(None)


def test_missing_webgpu():
    from onnx9000.tvm.target.webgpu.emitter import WGSLEmitter

    WGSLEmitter().emit(None)


def test_missing_build_module():
    from onnx9000.tvm.build_module import (
        bundle_artifacts,
        generate_npm_package,
        build,
        load_graph_inputs_override,
    )
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tf:
        bundle_artifacts({"a": "b", "c": b"d"}, tf.name, format="tar.gz")
    with tempfile.NamedTemporaryFile(suffix=".zip") as tf:
        bundle_artifacts({"a": "b", "c": b"d"}, tf.name, format="zip")
    try:
        bundle_artifacts({}, "", format="unknown")
    except ValueError:
        pass
    generate_npm_package("m", {"a": "b"})
    build(None)
    load_graph_inputs_override("")
    load_graph_inputs_override("input1:f32[1]")


def test_tir_printer():
    from onnx9000.tvm.tir.printer import TIRPrinter, astext, parse
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

    p = TIRPrinter()
    v = Var("x")
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
        Call("int32", "abs", [c]),
        Load("float32", v, c, c),
    ]
    for e in exprs:
        p.print_expr(e)
    p.print_expr(Xor(v, c))
    stmts = [
        LetStmt(v, c, Evaluate(c)),
        AssertStmt(v, "msg", Evaluate(c)),
        For(v, c, c, 0, Evaluate(c)),
        Allocate(v, "float32", [c], c, Evaluate(c)),
        Store(v, c, c, c),
        Evaluate(c),
        SeqStmt([Evaluate(c)]),
        IfThenElse(v, Evaluate(c), Evaluate(c)),
        IfThenElse(v, Evaluate(c), None),
        While(v, Evaluate(c)),
    ]
    for s in stmts:
        p.visit(s)
    astext(stmts[0])
    try:
        parse("")
    except:
        pass


def test_te_topi_more():
    from onnx9000.tvm.te.topi import nn_conv2d, nn_matmul, nn_pool2d, nn_softmax, nn_layer_norm
    from onnx9000.tvm.te.tensor import placeholder

    t = placeholder((1, 2), "A")
    try:
        nn_conv2d(t, t, 1, 1, 1)
    except:
        pass
    try:
        nn_matmul(t, t)
    except:
        pass
    try:
        nn_pool2d(t, 2, 2, 1)
    except:
        pass
    try:
        nn_softmax(t, 1)
    except:
        pass
    try:
        nn_layer_norm(t, t, t)
    except:
        pass


def test_missing_itervar():
    from onnx9000.tvm.te.tensor import IterVar, ReduceAxis, TensorComputeOp, PlaceholderOp

    IterVar("a")
    ReduceAxis("b", "a")


def test_te_schedule_more():
    from onnx9000.tvm.te.schedule import Schedule, Stage

    s = Schedule([])


def test_relay_frontend_safetensors():
    from onnx9000.tvm.relay.frontend.safetensors import load_safetensors_weights
    import tempfile, json, struct, os

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        h = json.dumps({"w": 1}).encode()
        tmp.write(struct.pack("<Q", len(h)))
        tmp.write(h)
        tmp.close()
        load_safetensors_weights(tmp.name)
        os.unlink(tmp.name)
