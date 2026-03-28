"""Module providing functionality for test_hub_extra."""

import os
from unittest.mock import patch, MagicMock
from onnx9000.toolkit.safetensors.hub import cached_download


@patch("onnx9000.toolkit.safetensors.hub.urlopen")
def test_download_from_hub_with_token(mock_urlopen):
    """Test download from hub with token."""
    mock_response = MagicMock()
    mock_response.read.return_value = b""
    mock_urlopen.return_value.__enter__.return_value = mock_response

    with patch.dict(os.environ, {"HF_TOKEN": "secret_token"}):
        with patch("onnx9000.toolkit.safetensors.hub.os.path.exists", return_value=False):
            # Also mock open to avoid actually trying to write a file
            with patch("onnx9000.toolkit.safetensors.hub.open", MagicMock()):
                cached_download("http://fake.url/model.safetensors")
