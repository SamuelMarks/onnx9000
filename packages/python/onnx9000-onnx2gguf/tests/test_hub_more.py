"""Unit tests for onnx2gguf Hugging Face hub configuration fetcher."""

from unittest.mock import patch

from onnx9000.onnx2gguf.hub import fetch_hf_config


@patch("urllib.request.urlopen")
def test_fetch_hf_config_exception(mock_urlopen: object) -> None:
    """Verify fetch_hf_config returns None when urlopen raises a generic error."""
    mock_urlopen.side_effect = RuntimeError("Generic error")  # type: ignore[attr-defined]
    res = fetch_hf_config("test/model")
    assert res is None
