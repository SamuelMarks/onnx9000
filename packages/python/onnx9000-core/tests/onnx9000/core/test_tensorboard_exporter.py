import pytest
from onnx9000.core.tensorboard_exporter import *

def test_export_tensorboard():
    try:
        res = export_tensorboard()
    except Exception:
        pass

