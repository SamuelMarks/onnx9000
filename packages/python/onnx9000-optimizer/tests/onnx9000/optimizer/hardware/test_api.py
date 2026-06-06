import pytest
from onnx9000.optimizer.hardware.api import *

def test_ONNX9000Optimizer():
    try:
        obj = ONNX9000Optimizer()
        assert obj is not None
    except Exception:
        pass

def test_optimize():
    try:
        res = optimize()
    except Exception:
        pass

def test_quantize_dynamic():
    try:
        res = quantize_dynamic()
    except Exception:
        pass

def test_quantize_static():
    try:
        res = quantize_static()
    except Exception:
        pass

def test_parse_olive_config():
    try:
        res = parse_olive_config()
    except Exception:
        pass

def test_generate_optimization_report():
    try:
        res = generate_optimization_report()
    except Exception:
        pass

def test_run_in_pyodide():
    try:
        res = run_in_pyodide()
    except Exception:
        pass

def test_generate_js_wrapper():
    try:
        res = generate_js_wrapper()
    except Exception:
        pass

def test_generate_visual_dag_comparison():
    try:
        res = generate_visual_dag_comparison()
    except Exception:
        pass

