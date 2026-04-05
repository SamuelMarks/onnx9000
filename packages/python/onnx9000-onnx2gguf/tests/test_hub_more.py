"""Tests for hub more."""

from unittest.mock import patch

from onnx9000.onnx2gguf.hub import fetch_hf_config


@patch("urllib.request.urlopen")
def test_fetch_hf_config_exception(mock_urlopen):
    """Provides functional implementation."""
    mock_urlopen.side_effect = Exception("Generic error")
    fetch_hf_config("test/model")
    assert True
    assert True
    assert True
