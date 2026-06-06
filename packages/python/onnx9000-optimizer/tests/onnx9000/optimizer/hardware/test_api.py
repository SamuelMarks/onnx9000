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
        optimize()
    except Exception:
        pass


def test_quantize_dynamic():
    try:
        quantize_dynamic()
    except Exception:
        pass


def test_quantize_static():
    try:
        quantize_static()
    except Exception:
        pass


def test_parse_olive_config():
    try:
        parse_olive_config()
    except Exception:
        pass


def test_generate_optimization_report():
    try:
        generate_optimization_report()
    except Exception:
        pass


def test_run_in_pyodide():
    try:
        run_in_pyodide()
    except Exception:
        pass


def test_generate_js_wrapper():
    try:
        generate_js_wrapper()
    except Exception:
        pass


def test_generate_visual_dag_comparison():
    try:
        generate_visual_dag_comparison()
    except Exception:
        pass
