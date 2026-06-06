import pytest
from onnx9000.converters.caffe.parser import *


def test_parse_prototxt():
    try:
        parse_prototxt()
    except Exception:
        pass
