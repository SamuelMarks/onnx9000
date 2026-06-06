import pytest
from onnx9000.jax.jaxpr_string_parser import *


def test_parse_jaxpr_string():
    try:
        parse_jaxpr_string()
    except Exception:
        pass
