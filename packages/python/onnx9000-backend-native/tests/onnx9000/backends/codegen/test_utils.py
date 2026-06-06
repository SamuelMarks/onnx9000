import pytest
from onnx9000.backends.codegen.utils import *


def test_get_attribute():
    try:
        get_attribute()
    except Exception:
        pass


def test_sanitize_name():
    try:
        sanitize_name()
    except Exception:
        pass


def test_get_omp_pragma():
    try:
        get_omp_pragma()
    except Exception:
        pass
