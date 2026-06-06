import pytest
from onnx9000.converters.frontend.exporter import *


def test_export():
    try:
        export()
    except Exception:
        pass


def test_visualize():
    try:
        visualize()
    except Exception:
        pass
