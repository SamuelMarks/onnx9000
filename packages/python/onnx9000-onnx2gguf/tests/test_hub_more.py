import pytest
from onnx9000.onnx2gguf.hub import fetch_hf_config
from unittest.mock import patch
import urllib.error


@patch("urllib.request.urlopen")
def test_fetch_hf_config_exception(mock_urlopen):
    mock_urlopen.side_effect = Exception("Generic error")
    config, tokenizer, url = fetch_hf_config("test/model")
    assert config == {}
    assert tokenizer == ""
    assert url == "https://huggingface.co/test/model"
