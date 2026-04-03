import os
import pytest
from unittest.mock import patch, MagicMock
from onnx9000_diffusers.utils import fetch_hub_file


def test_fetch_hub_file_success(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()

    mock_resp = MagicMock()
    mock_resp.read.return_value = b""

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        path = fetch_hub_file("repo/id", "test_file.txt", str(cache))
        assert os.path.exists(path)


def test_fetch_hub_file_error_removes(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()

    def fail_copy(*args, **kwargs):
        raise Exception("Mock Error")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        with patch("shutil.copyfileobj", side_effect=fail_copy):
            with pytest.raises(Exception, match="Mock Error"):
                fetch_hub_file("repo/id", "test_file2.txt", str(cache))

    assert not os.path.exists(cache / "repo--id" / "test_file2.txt")
