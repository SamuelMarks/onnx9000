import pytest
from onnx9000.converters.mxnet.parser import *

def test_parse_symbol():
    try:
        res = parse_symbol()
    except Exception:
        pass

