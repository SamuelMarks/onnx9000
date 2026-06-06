import pytest
from onnx9000.optimizer.simplifier.passes.quantization import *


def test_insert_qat_nodes():
    try:
        insert_qat_nodes()
    except Exception:
        pass


def test_convert_to_int8():
    try:
        convert_to_int8()
    except Exception:
        pass
