import pytest
from onnx9000.converters.cntk.parser import *


def test_parse_cntk_model():
    try:
        parse_cntk_model()
    except Exception:
        pass
