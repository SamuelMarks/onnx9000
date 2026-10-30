"""Tests for extras."""

import urllib.error
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.onnx2gguf.hub import fetch_hf_config
from onnx9000.onnx2gguf.reverse import reconstruct_onnx


def test_hub_fetch_tokenizer_exception() -> None:
    """Verify that hub fetch tokenizer exception behaves as expected.

    Returns:
        Result of test hub fetch tokenizer exception.

    Raises:
        Exception: If an invalid state or argument is encountered during execution.
    """

    def mock_urlopen(
        req: int | float | str | bool | bytes | list | dict | tuple | None,
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Mock urlopen."""
        if "config.json" in req.full_url:
            res = MagicMock()
            res.__enter__.return_value.read.return_value = b"{}"
            return res
        else:
            raise Onnx9000Error("Mock error")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        assert fetch_hf_config("test/model") is None


def test_hub_fetch_tokenizer_httperror() -> None:
    """Verify that hub fetch tokenizer httperror behaves as expected.

    Returns:
        Result of test hub fetch tokenizer httperror.
    """

    def mock_urlopen(
        req: int | float | str | bool | bytes | list | dict | tuple | None,
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Mock urlopen."""
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


def test_gguf2onnx_split() -> None:
    """Verify that gguf2onnx split behaves as expected."""

    class MockReader:
        """Mock reader."""

        def __init__(self) -> None:
            """Initialize MockReader with configured parameters and state."""
            self.kvs = {"split.index": 1, "tokenizer.ggml.tokens": []}
            self.tensors = {}

    assert reconstruct_onnx(MockReader()) is None


def test_gguf2onnx_success() -> None:
    """Verify that gguf2onnx success behaves as expected."""

    class MockReader2:
        """Mock reader 2."""

        def __init__(self) -> None:
            """Initialize MockReader2 with configured parameters and state."""
            self.kvs = {"general.name": "success"}
            self.tensors = {}

    g = reconstruct_onnx(MockReader2())
    assert g is not None
    assert g.name == "success"
