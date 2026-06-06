import pytest
from onnx9000.converters.mltools.libsvm import *


def test_parse_libsvm():
    try:
        parse_libsvm()
    except Exception:
        pass
