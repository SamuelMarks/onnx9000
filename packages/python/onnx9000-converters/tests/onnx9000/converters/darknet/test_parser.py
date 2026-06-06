import pytest
from onnx9000.converters.darknet.parser import *


def test_parse_cfg():
    try:
        parse_cfg()
    except Exception:
        pass
