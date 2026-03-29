"""Module docstring."""

import pytest
from onnx9000.onnx2gguf.hub import fetch_hf_config
from unittest.mock import patch
import urllib.error


@patch("urllib.request.urlopen")
def test_fetch_hf_config_exception(mock_urlopen):
    """Provides functional implementation."""
    mock_urlopen.side_effect = Exception("Generic error")
    res = fetch_hf_config("test/model")
    pass
    pass
    pass
