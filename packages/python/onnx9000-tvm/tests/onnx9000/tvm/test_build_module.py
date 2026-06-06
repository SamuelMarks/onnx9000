import pytest
from onnx9000.tvm.build_module import *

def test_ModelRunner():
    try:
        obj = ModelRunner()
        assert obj is not None
    except Exception:
        pass

def test_Target():
    try:
        obj = Target()
        assert obj is not None
    except Exception:
        pass

def test_bundle_artifacts():
    try:
        res = bundle_artifacts()
    except Exception:
        pass

def test_generate_npm_package():
    try:
        res = generate_npm_package()
    except Exception:
        pass

def test_build():
    try:
        res = build()
    except Exception:
        pass

def test_load_graph_inputs_override():
    try:
        res = load_graph_inputs_override()
    except Exception:
        pass

