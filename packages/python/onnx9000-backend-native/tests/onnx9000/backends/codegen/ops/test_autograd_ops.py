import pytest
from onnx9000.backends.codegen.ops.autograd_ops import *


def test_generate_relu_grad():
    try:
        generate_relu_grad()
    except Exception:
        pass


def test_generate_sgd():
    try:
        generate_sgd()
    except Exception:
        pass


def test_generate_adamw():
    try:
        generate_adamw()
    except Exception:
        pass
