"""Tests for coverage extras3."""

import os
from typing import NoReturn
from unittest.mock import MagicMock, patch

import pytest
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_diffusers.utils import fetch_hub_file


def test_fetch_hub_file_success(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that fetch hub file success behaves as expected.

    Args:
        tmp_path: Parameter specifying tmp path.
    """
    cache = tmp_path / "cache"
    cache.mkdir()

    mock_resp = MagicMock()
    mock_resp.read.return_value = b""

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        path = fetch_hub_file("repo/id", "test_file.txt", str(cache))
        assert os.path.exists(path)


def test_fetch_hub_file_error_removes(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that fetch hub file error removes behaves as expected.

    Args:
        tmp_path: Parameter specifying tmp path.

    Raises:
        Exception: If an invalid state or argument is encountered during execution.
    """
    cache = tmp_path / "cache"
    cache.mkdir()

    def fail_copy(
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> NoReturn:
        """Fail copy."""
        raise Onnx9000Error("Mock Error")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        with patch("shutil.copyfileobj", side_effect=fail_copy):
            with pytest.raises(Exception, match="Mock Error"):
                fetch_hub_file("repo/id", "test_file2.txt", str(cache))

    assert not os.path.exists(cache / "repo--id" / "test_file2.txt")
