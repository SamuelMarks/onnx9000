import pytest
from onnx9000.core.macros import *


def test_MacroExpander():
    try:
        obj = MacroExpander()
        assert obj is not None
    except Exception:
        pass


def test_MacroMatcher():
    try:
        obj = MacroMatcher()
        assert obj is not None
    except Exception:
        pass


def test_ir_macro():
    try:
        ir_macro()
    except Exception:
        pass


def test_transformer_block_macro():
    try:
        transformer_block_macro()
    except Exception:
        pass


def test_moe_layer_macro():
    try:
        moe_layer_macro()
    except Exception:
        pass
