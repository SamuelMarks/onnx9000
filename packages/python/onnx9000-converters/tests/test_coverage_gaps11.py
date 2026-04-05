"""Tests for coverage gaps11."""

import pytest
from onnx9000.converters.flax_parser import parse_msgpack
from onnx9000.converters.jax.jaxpr_string_parser import parse_jaxpr_string


def test_flax_parser_gaps():
    """Docstring for D103."""
    with pytest.raises(ValueError, match="Unexpected end of data"):
        parse_msgpack(bytes([0xD9]))

    with pytest.raises(ValueError, match="Unexpected end of data"):
        parse_msgpack(bytes([0xC7]))


def test_jaxpr_string_parser_gap():
    """Docstring for D103."""
    jaxpr = """{ lambda ; a:f32[]. let

    b:f32[] = test_op[param=foo,] a
  in (b,) }"""
    parsed = parse_jaxpr_string(jaxpr)
    assert parsed["eqns"][0]["params"]["param"] == "foo"
