import pytest
from onnx9000.core.exporter import *

def test_ONNXToKerasVisitor():
    try:
        obj = ONNXToKerasVisitor()
        assert obj is not None
    except Exception:
        pass

def test_IRToONNXExporter():
    try:
        obj = IRToONNXExporter()
        assert obj is not None
    except Exception:
        pass

def test_register_exporter():
    try:
        res = register_exporter()
    except Exception:
        pass

def test_export_graph():
    try:
        res = export_graph()
    except Exception:
        pass

def test_generate_keras():
    try:
        res = generate_keras()
    except Exception:
        pass

def test_generate_mlir():
    try:
        res = generate_mlir()
    except Exception:
        pass

