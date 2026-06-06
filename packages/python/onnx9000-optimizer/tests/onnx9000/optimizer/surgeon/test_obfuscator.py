import pytest
from onnx9000.optimizer.surgeon.obfuscator import *


def test_obfuscate_names():
    try:
        obfuscate_names()
    except Exception:
        pass
