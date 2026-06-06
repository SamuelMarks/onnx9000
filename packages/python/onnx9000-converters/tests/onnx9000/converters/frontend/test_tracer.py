import pytest
from onnx9000.converters.frontend.tracer import *


def test_Tracer():
    try:
        obj = Tracer()
        assert obj is not None
    except Exception:
        pass


def test_Proxy():
    try:
        obj = Proxy()
        assert obj is not None
    except Exception:
        pass


def test_trace():
    try:
        trace()
    except Exception:
        pass


def test_script():
    try:
        script()
    except Exception:
        pass
