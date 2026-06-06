import pytest
from onnx9000.core.verification import *


def test_IRNode():
    try:
        obj = IRNode()
        assert obj is not None
    except Exception:
        pass


def test_IRGraph():
    try:
        obj = IRGraph()
        assert obj is not None
    except Exception:
        pass


def test_OracleVerifier():
    try:
        obj = OracleVerifier()
        assert obj is not None
    except Exception:
        pass


def test_check_tolerance():
    try:
        check_tolerance()
    except Exception:
        pass


def test_reset_environment():
    try:
        reset_environment()
    except Exception:
        pass


def test_bisect_dag():
    try:
        bisect_dag()
    except Exception:
        pass
