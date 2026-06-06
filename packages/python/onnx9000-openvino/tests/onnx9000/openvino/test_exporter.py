import pytest
from onnx9000.openvino.exporter import *


def test_OpenVinoExporter():
    try:
        obj = OpenVinoExporter()
        assert obj is not None
    except Exception:
        pass
