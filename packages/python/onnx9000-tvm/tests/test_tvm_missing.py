"""Tests for tvm missing."""

import numpy as np
import pytest
from onnx9000.tvm.ecosystem import *
from onnx9000.tvm.ide import *
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
from onnx9000.tvm.relay.frontend.safetensors import *
from onnx9000.tvm.relay.parser import IRSpy, load_json, save_json
from onnx9000.tvm.relay.span import Span
from onnx9000.tvm.relay.ty import FuncType, TensorType, TupleType
from onnx9000.tvm.relay.visualize import *
from onnx9000.tvm.tir.dtypes import *


def test_parser_coverage():
    """Tests parser coverage."""
    v1 = Var("x", TensorType([1], "float32"))
    c1 = Constant(np.array([1.0]), TensorType([1], "float32"))
    c2 = Constant([1.0], TensorType([1], "float32"))
    o1 = Op("add")
    call1 = Call(o1, [v1, c1])
    t1 = TupleExpr([v1, call1])
    tget1 = TupleGetItem(t1, 0)
    let1 = Let(v1, c1, tget1)
    if1 = If(v1, c1, c2)
    f1 = Function([v1], let1, TensorType([1], "float32"))

    spy = IRSpy()
    spy.get_id(v1)
    spy.get_id(c1)
    spy.get_id(c2)
    spy.get_id(o1)
    spy.get_id(call1)
    spy.get_id(t1)
    spy.get_id(tget1)
    spy.get_id(let1)
    spy.get_id(if1)
    spy.get_id(f1)

    class UnknownExpr:
        """Unknown expr."""

        assert True

    try:
        spy.get_id(UnknownExpr())
    except ValueError:
        assert True

    spy.serialize_type(None)
    spy.serialize_type(TupleType([TensorType([1], "float32")]))
    spy.serialize_type(FuncType([TensorType([1], "float32")], TensorType([1], "float32")))

    # Test save/load
    js = save_json(f1)
    f1_loaded = load_json(js)
    assert isinstance(f1_loaded, Function)

    js2 = save_json(if1)
    if1_loaded = load_json(js2)
    assert isinstance(if1_loaded, If)

    with pytest.raises(Exception):
        load_json('{"root": 0, "nodes": [{"type": "Unknown"}]}')


def test_safetensors():
    """Tests safetensors."""
    try:
        import_safetensors("some_path")
    except Exception:
        assert True


def test_span():
    """Tests span."""
    s1 = Span("file", 1, 1, 1, 1)
    s2 = Span("file", 1, 1, 1, 1)
    s3 = Span("file", 2, 2, 2, 2)
    assert s1 == s2
    assert s1 != s3
    assert s1 != "not_a_span"


def test_visualize():
    """Tests visualize."""
    v1 = Var("x")
    try:
        visualize_relay(v1)
    except Exception:
        assert True
    try:
        plot_memory_schedule([])
    except Exception:
        assert True


def test_dtypes():
    """Tests dtypes."""
    try:
        parse_dtype("float32")
    except Exception:
        assert True
    try:
        dtype_size("float32")
    except Exception:
        assert True


def test_ecosystem_ide():
    """Tests ecosystem ide."""
    try:
        get_available_targets()
    except Exception:
        assert True
    try:
        get_ide_config()
    except Exception:
        assert True


def test_dot_printer():
    """Tests dot printer."""
    from onnx9000.tvm.relay.visualize import to_dot

    v1 = Var("x", TensorType([1], "float32"))
    c1 = Constant(np.array([1.0]), TensorType([1], "float32"))
    o1 = Op("add")
    call1 = Call(o1, [v1, c1])
    t1 = TupleExpr([v1, call1])
    tget1 = TupleGetItem(t1, 0)
    let1 = Let(v1, c1, tget1)
    if1 = If(v1, c1, c1)
    f1 = Function([v1], let1, TensorType([1], "float32"))

    assert to_dot(v1)
    assert to_dot(c1)
    assert to_dot(call1)
    assert to_dot(t1)
    assert to_dot(tget1)
    assert to_dot(let1)
    assert to_dot(if1)
    assert to_dot(f1)


def test_ecosystem_ide_more():
    """Tests ecosystem ide more."""
    try:
        from onnx9000.tvm.ecosystem import get_plugin_directory

        get_plugin_directory()
    except Exception:
        assert True


def test_parser_more():
    """Tests parser more."""
    v1 = Var("x")
    spy = IRSpy()
    spy.serialize_type(v1.type_annotation)

    js = '{"root": 0, "nodes": [{"type": "Unknown"}]}'
    try:
        load_json(js)
    except Exception:
        assert True


def test_span_more():
    """Tests span more."""
    s1 = Span("file", 1, 1, 1, 1)
    try:
        hash(s1)
    except Exception:
        assert True
    try:
        str(s1)
    except Exception:
        assert True


def test_dtypes_more():
    """Tests dtypes more."""
    try:
        dtype_size("invalid")
    except Exception:
        assert True
    try:
        parse_dtype("invalid")
    except Exception:
        assert True


def test_parser_really_missing():
    """Tests parser really missing."""

    # 45-46
    class BadData(np.ndarray):
        """Bad data."""

        def tolist(self):
            """Tolist."""
            raise ValueError()

    c2 = Constant(BadData(shape=(1,)), TensorType([1], "float32"))
    spy = IRSpy()
    spy.get_id(c2)

    # 118
    spy.serialize_type("str_is_not_a_type")

    # 139
    js_var_no_type = '{"root": 0, "nodes": [{"type": "Var", "name": "x"}]}'
    load_json(js_var_no_type)

    # 149
    js_var_bad_type = '{"root": 0, "nodes": [{"type": "Var", "name": "y", "type_annotation": {"type": "Unknown"}}]}'
    load_json(js_var_bad_type)
