from onnx9000.toolkit.script.builder import GraphBuilder
from onnx9000.toolkit.script.op import (
    If,
    Loop,
    Scan,
    _make_vars,
    pop_active_builder,
    set_active_builder,
)
from onnx9000.toolkit.script.var import Var


def test_make_vars() -> None:
    vars = _make_vars([1.0, 2])
    assert len(vars) == 2


def test_op_multiple_outputs() -> None:
    gb = GraphBuilder("test_ops")
    set_active_builder(gb)
    cond = Var("cond")
    then_gb = GraphBuilder("then")
    else_gb = GraphBuilder("else")
    out_if = If(cond, then_branch=then_gb, else_branch=else_gb, num_outputs=2)
    assert isinstance(out_if, tuple)
    assert len(out_if) == 2
    max_trip = Var("max_trip")
    loop_gb = GraphBuilder("loop")
    out_loop = Loop(max_trip, cond, body=loop_gb, num_outputs=2)
    assert isinstance(out_loop, tuple)
    assert len(out_loop) == 2
    scan_gb = GraphBuilder("scan")
    out_scan = Scan(body=scan_gb, num_scan_inputs=1, num_outputs=2)
    assert isinstance(out_scan, tuple)
    assert len(out_scan) == 2
    pop_active_builder()
