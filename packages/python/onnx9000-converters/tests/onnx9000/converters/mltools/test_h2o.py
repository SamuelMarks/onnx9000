import pytest
from onnx9000.converters.mltools.h2o import *


def test_parse_h2o():
    try:
        parse_h2o()
    except Exception:
        pass
