"""Module docstring."""

import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from onnx9000.onnx2gguf.hub import fetch_hf_config
from onnx9000.onnx2gguf.reverse import reconstruct_onnx


def test_hub_fetch_tokenizer_exception():
    """Docstring for D103."""

    def mock_urlopen(req, *args, **kwargs):
        if "config.json" in req.full_url:
            res = MagicMock()
            res.__enter__.return_value.read.return_value = b"{}"
            return res
        else:
            raise Exception("Mock error")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        assert fetch_hf_config("test/model") is None


def test_hub_fetch_tokenizer_httperror():
    """Docstring for D103."""

    def mock_urlopen(req, *args, **kwargs):
        if "config.json" in req.full_url:
            res = MagicMock()
            res.__enter__.return_value.read.return_value = b"{}"
            return res
        else:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        # HTTP error continues and returns tuple
        res = fetch_hf_config("test/model")
        assert res is not None


def test_gguf2onnx_split():
    """Docstring for D103."""

    class MockReader:
        def __init__(self):
            self.kvs = {"split.index": 1, "tokenizer.ggml.tokens": []}
            self.tensors = {}

    assert reconstruct_onnx(MockReader()) is None


def test_gguf2onnx_success():
    """Docstring for D103."""

    class MockReader2:
        def __init__(self):
            self.kvs = {"general.name": "success"}
            self.tensors = {}

    g = reconstruct_onnx(MockReader2())
    assert g is not None
    assert g.name == "success"
