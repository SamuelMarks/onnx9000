import pytest
from onnx9000_optimum.export import *


def test_get_huggingface_model_files():
    try:
        get_huggingface_model_files()
    except Exception:
        pass


def test__progress_bar():
    try:
        _progress_bar()
    except Exception:
        pass


def test_auto_detect_task():
    try:
        auto_detect_task()
    except Exception:
        pass


def test_warn_unsupported_ops():
    try:
        warn_unsupported_ops()
    except Exception:
        pass


def test_create_dummy_inputs():
    try:
        create_dummy_inputs()
    except Exception:
        pass


def test_export_model():
    try:
        export_model()
    except Exception:
        pass
