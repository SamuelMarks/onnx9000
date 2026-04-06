"""Tests for tvm missing final."""

import pytest
from onnx9000.tvm.ecosystem import TVMParityCertifier
from onnx9000.tvm.relay.expr import Constant, Var
from onnx9000.tvm.relay.parser import IRSpy, load_json, save_json
from onnx9000.tvm.relay.span import Span, set_span
from onnx9000.tvm.relay.ty import TensorType
from onnx9000.tvm.tir.dtypes import is_supported


def test_the_rest_of_missing():
    """Test docstring."""
    # ecosystem 15
    TVMParityCertifier().certify()

    # parser
    c1 = Constant("invalid_data", TensorType([1], "float32"))
    spy = IRSpy()
    spy.get_id(c1)

    # parse_type
    js2 = '{"root": 0, "nodes": [{"type": "Var", "name": "a", "type_annotation": {"type": "TupleType", "fields": [{"type": "TensorType", "shape": [1], "dtype": "float32"}]}}]}'
    load_json(js2)
    js3 = '{"root": 0, "nodes": [{"type": "Var", "name": "a", "type_annotation": {"type": "FuncType", "arg_types": [{"type": "TensorType", "shape": [1], "dtype": "float32"}], "ret_type": {"type": "TensorType", "shape": [1], "dtype": "float32"}}}]}'
    load_json(js3)

    # span 20-21
    s = Span("a", 1, 1, 1, 1)
    set_span(Var("a"), s)

    # dtypes 29
    is_supported("float32")
    is_supported("invalid")
