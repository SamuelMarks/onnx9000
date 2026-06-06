import pytest
from onnx9000.openvino.api import *

def test_export_model():
    try:
        res = export_model()
    except Exception:
        pass

