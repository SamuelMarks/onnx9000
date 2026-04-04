"""Module docstring."""


def test_script_op_make_vars():
    """Docstring for D103."""
    from onnx9000.toolkit.script.op import _make_vars
    from onnx9000.toolkit.script.var import Var

    # 46 is `return [_make_var(v) for v in vals]`
    # Did we never test _make_vars?
    res = _make_vars([1, 2])
    assert len(res) == 2
    assert isinstance(res[0], Var)


def test_script_op_builder():
    """Docstring for D103."""
    import onnx9000.core.ir as ir
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test")
    with b:
        # 60: for a in arg: inputs.append(_make_var(a))
        # this is for `if isinstance(arg, list):`
        # 66, 71-72
        # `if op_type in ["TopK", "Split", "LSTM"] and op_type == "TopK": num_outputs = 2` -> wait! I can just use op.TopK!
        # `if op_type == "Squeeze" and "axes" in kwargs:`
        op.TopK([1, 2], k=1)
        op.Squeeze(1, axes=[0])

        # 98-104 is for `if num_outputs == 1: return out_vars[0]` else `return tuple(out_vars)`
        # Wait, if `num_outputs == 2` it returns a tuple, which hits 104 `return tuple(out_vars)`.
        # What about `out_vars = tuple(out_vars)`?


def test_script_op_constant():
    """Docstring for D103."""
    import numpy as np
    import pytest
    from onnx9000.toolkit.script.op import Constant

    # array (93)
    Constant(np.array([1]))
    # float (95)
    Constant(1.0)
    # int (97) - wait, already hit maybe?
    Constant(1)

    # 102-104: list of non-ints, and ValueError
    Constant([1.0, 2.0])
    with pytest.raises(ValueError):
        Constant("bad")


def test_script_op_if_loop():
    """Docstring for D103."""
    from onnx9000.toolkit.script.op import If, Loop
    from onnx9000.toolkit.script.var import Var

    # 117-133 is If
    # 140-157 is Loop
    # 164-179 is Scan
    res1 = If(1, "then", "else", 0)
    assert res1 is None
    res2 = If(1, "then", "else", 1)
    assert isinstance(res2, Var)
    res3 = If(1, "then", "else", 2)
    assert isinstance(res3, tuple)


def test_script_op_loop_scan():
    """Docstring for D103."""
    from onnx9000.toolkit.script.op import Loop, Scan

    res1 = Loop(10, 1, "body", 0)
    assert res1 is None
    Loop(10, 1, "body", 1)
    res3 = Loop(10, 1, "body", 2)
    assert isinstance(res3, tuple)

    res4 = Scan([], "body", 0)
    assert res4 is None
    res5 = Scan([], "body", 1)
    assert not isinstance(res5, tuple)
    res6 = Scan([], "body", 2)
    assert isinstance(res6, tuple)


def test_script_op_concat_list():
    """Docstring for D103."""
    from onnx9000.toolkit.script.op import op

    # 60-61
    from onnx9000.toolkit.script.schema import set_target_opset

    set_target_opset(18)
    op.Concat([1, 2], axis=0)


def test_script_op_scan_1():
    """Docstring for D103."""
    from onnx9000.toolkit.script.op import Scan

    res = Scan([], "body", 1)
    assert res is not None
    # 174 is `return out_vars[0]`


def test_script_op_scan_num_outputs_1():
    """Docstring for D103."""
    from onnx9000.toolkit.script.op import Scan

    res = Scan([], "body", num_outputs=1)
    assert not isinstance(res, tuple)


def test_script_op_scan_num_outputs_1_again():
    """Docstring for D103."""
    from onnx9000.toolkit.script.op import Scan
    from onnx9000.toolkit.script.var import Var

    res = Scan([], "body", num_outputs=1)
    assert isinstance(res, Var)
    assert res.name is not None
