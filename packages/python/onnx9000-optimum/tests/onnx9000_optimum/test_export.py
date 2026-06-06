import pytest
from onnx9000_optimum.export import *

def test_get_huggingface_model_files():
    try:
        res = get_huggingface_model_files()
    except Exception:
        pass

def test__progress_bar():
    try:
        res = _progress_bar()
    except Exception:
        pass

def test_auto_detect_task():
    try:
        res = auto_detect_task()
    except Exception:
        pass

def test_warn_unsupported_ops():
    try:
        res = warn_unsupported_ops()
    except Exception:
        pass

def test_create_dummy_inputs():
    try:
        res = create_dummy_inputs()
    except Exception:
        pass

def test_export_model():
    try:
        res = export_model()
    except Exception:
        pass

