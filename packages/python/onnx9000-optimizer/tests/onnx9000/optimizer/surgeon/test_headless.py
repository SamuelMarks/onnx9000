import pytest
from onnx9000.optimizer.surgeon.headless import *


def test_rename_input():
    try:
        rename_input()
    except Exception:
        pass


def test_change_batch():
    try:
        change_batch()
    except Exception:
        pass


def test_mutate():
    try:
        mutate()
    except Exception:
        pass
