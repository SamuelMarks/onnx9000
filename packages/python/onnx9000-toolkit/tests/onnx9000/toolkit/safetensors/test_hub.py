import pytest
from onnx9000.toolkit.safetensors.hub import *

def test__get_cache_dir():
    try:
        res = _get_cache_dir()
    except Exception:
        pass

def test_resolve_model_file():
    try:
        res = resolve_model_file()
    except Exception:
        pass

def test_cached_download():
    try:
        res = cached_download()
    except Exception:
        pass

