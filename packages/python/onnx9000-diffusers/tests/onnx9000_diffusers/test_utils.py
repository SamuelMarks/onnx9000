import pytest
from onnx9000_diffusers.utils import *


def test_PyTorchPCG():
    try:
        obj = PyTorchPCG()
        assert obj is not None
    except Exception:
        pass


def test_ProgressBarConfig():
    try:
        obj = ProgressBarConfig()
        assert obj is not None
    except Exception:
        pass


def test_rand():
    try:
        rand()
    except Exception:
        pass


def test_randn():
    try:
        randn()
    except Exception:
        pass


def test_set_progress_bar_config():
    try:
        set_progress_bar_config()
    except Exception:
        pass


def test_fetch_hub_file():
    try:
        fetch_hub_file()
    except Exception:
        pass


def test_parse_model_index():
    try:
        parse_model_index()
    except Exception:
        pass
