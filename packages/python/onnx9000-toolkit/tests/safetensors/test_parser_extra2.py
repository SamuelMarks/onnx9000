"""Tests for parser extra2."""

from typing import NoReturn
from unittest.mock import patch

import pytest
from onnx9000.toolkit.safetensors.hub import cached_download, resolve_model_file
from onnx9000.toolkit.safetensors.parser import SafeTensors


def test_hub_extra() -> None:
    """Verify that hub extra behaves as expected.

    Raises:
        HTTPError: If an invalid state or argument is encountered during execution.
    """

    # 44-45
    def mock_urlopen(
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> NoReturn:
        """Mock urlopen."""
        from urllib.error import HTTPError

        raise HTTPError("url", 404, "Not Found", {}, None)

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        assert resolve_model_file("foo/bar") is None

    # 73
    assert cached_download("https://huggingface.co/user/repo/blob/main/file") is None


def test_get_onnx9000_tensor() -> None:
    """Verify that get onnx9000 tensor behaves as expected."""
    parser = SafeTensors.__new__(SafeTensors)
    parser.tensors = {
        "t1": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "t2": {"dtype": "UNKN", "shape": [2, 2], "data_offsets": [0, 16]},
    }
    parser.get_tensor = lambda name: b"\x00" * 16

    t = parser.get_onnx9000_tensor("t1")
    assert t.name == "t1"

    from onnx9000.toolkit.safetensors.parser import SafetensorsInvalidDtypeError

    with pytest.raises(SafetensorsInvalidDtypeError):
        parser.get_onnx9000_tensor("t2")
