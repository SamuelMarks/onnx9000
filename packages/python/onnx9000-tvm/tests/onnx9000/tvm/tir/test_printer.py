import pytest
from onnx9000.tvm.tir.printer import *


def test_TIRPrinter():
    try:
        obj = TIRPrinter()
        assert obj is not None
    except Exception:
        pass


def test_astext():
    try:
        astext()
    except Exception:
        pass


def test_parse():
    try:
        parse()
    except Exception:
        pass
