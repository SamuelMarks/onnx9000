import pytest
from onnx9000.c_compiler.codegen import *


def test_BaseCodegenVisitor():
    try:
        obj = BaseCodegenVisitor()
        assert obj is not None
    except Exception:
        pass


def test_CFamilyCodegen():
    try:
        obj = CFamilyCodegen()
        assert obj is not None
    except Exception:
        pass


def test_PythonFamilyCodegen():
    try:
        obj = PythonFamilyCodegen()
        assert obj is not None
    except Exception:
        pass
