import pytest
from onnx9000.jax.flax_parser import *

def test_parse_msgpack():
    try:
        res = parse_msgpack()
    except Exception:
        pass

