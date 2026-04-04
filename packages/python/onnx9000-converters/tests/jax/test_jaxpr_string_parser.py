"""Module docstring."""

import pytest
from onnx9000.converters.jax.jaxpr_string_parser import parse_jaxpr_string


def test_parse_jaxpr_string():
    """Docstring for D103."""
    jaxpr = """{ lambda ; a:f32[10,10]. let
        b:f32[10,10] = dot_general[dimension_numbers=(((1,), (0,)), ((), ()))] a a
      in (b,) }"""

    parsed = parse_jaxpr_string(jaxpr)

    assert len(parsed["eqns"]) == 1
    eqn = parsed["eqns"][0]
    assert eqn["primitive"] == "dot_general"
    assert eqn["params"]["dimension_numbers"] == (((1,), (0,)), ((), ()))
    assert eqn["invars"][0]["name"] == "a"
    assert eqn["outvars"][0]["name"] == "b"


def test_parse_jaxpr_string_advanced():
    """Docstring for D103."""
    from onnx9000.converters.jax.jaxpr_string_parser import parse_jaxpr_string

    s = """
    { lambda ; a. let
        b:f32[] = add[ ] a 1
      in (b,) }
    """
    res = parse_jaxpr_string(s)
    assert len(res["eqns"]) == 1

    # Eval fallback
    s2 = """
    { lambda ; a. let
        b:f32[] = add[attr=invalid_python_syntax] a 1
      in (b,) }
    """
    res2 = parse_jaxpr_string(s2)
    assert res2["eqns"][0]["params"]["attr"] == "invalid_python_syntax"


def test_parse_jaxpr_string_empty():
    """Docstring for D103."""
    from onnx9000.converters.jax.jaxpr_string_parser import parse_jaxpr_string

    s = """
    
    """
    res = parse_jaxpr_string(s)
    assert len(res["eqns"]) == 0
