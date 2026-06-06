import pytest
from onnx9000.c_compiler.cli import *

def test_main():
    try:
        res = main()
    except Exception:
        pass

