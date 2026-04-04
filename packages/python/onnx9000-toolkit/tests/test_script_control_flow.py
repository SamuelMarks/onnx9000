"""Module docstring."""


def test_script_control_flow_all():
    """Docstring for D103."""
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.control_flow import (
        BranchContext,
        IfContextManager,
        LoopContextManager,
    )
    from onnx9000.toolkit.script.op import op
    from onnx9000.toolkit.script.var import Var

    # 15, 19-20, 24
    b = GraphBuilder("test_branch")
    bc = BranchContext(b)
    with bc:
        assert b.name == "test_branch"

    # 40, 44, 48-49: IfContextManager __enter__ / __exit__
    b_if = GraphBuilder("test_if_ctx")
    v = Var("cond")
    if_ctx = IfContextManager(b_if, v, 1)

    tb = if_ctx.Then()
    assert tb.builder.name == "test_if_ctx_then"
    eb = if_ctx.Else()
    assert eb.builder.name == "test_if_ctx_else"

    if_ctx.build()
    assert len(b_if.nodes) == 1
    assert b_if.nodes[0].op_type == "If"

    # 72, 76-77: LoopContextManager __enter__ / __exit__
    b_loop = GraphBuilder("test_loop_ctx")
    mtc = Var("mtc")
    cond2 = Var("cond2")
    loop_ctx = LoopContextManager(b_loop, mtc, cond2, 1)

    body_ctx = loop_ctx.Body()
    assert body_ctx.builder.name == "test_loop_ctx_loop_body"

    loop_ctx.build()
    assert len(b_loop.nodes) == 1
    assert b_loop.nodes[0].op_type == "Loop"
