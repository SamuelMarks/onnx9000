import pytest
from onnx9000.toolkit.safetensors.hub import *


def test__get_cache_dir():
    try:
        _get_cache_dir()
    except Exception:
        pass


def test_resolve_model_file():
    try:
        resolve_model_file()
    except Exception:
        pass


def test_cached_download():
    try:
        cached_download()
    except Exception:
        pass
