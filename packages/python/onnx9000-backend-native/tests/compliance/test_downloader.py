import io
import tempfile
import urllib.request
import zipfile
from pathlib import Path
import pytest
from onnx9000.backends.testing.downloader import download_and_extract_onnx_tests, get_node_test_dirs


def test_downloader_and_extractor(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        def mock_retrieve(url, filename):
            mem_zip = io.BytesIO()
            with zipfile.ZipFile(mem_zip, mode="w") as zf:
                z_info = zipfile.ZipInfo(
                    "onnx-1.14.0/onnx/backend/test/data/node/test_mock_node/model.onnx"
                )
                zf.writestr(z_info, b"mock_onnx_content")
            with open(filename, "wb") as f:
                f.write(mem_zip.getvalue())

        monkeypatch.setattr(urllib.request, "urlretrieve", mock_retrieve)
        extracted_dir = download_and_extract_onnx_tests(base_dir)
        assert extracted_dir.exists()
        node_dirs = get_node_test_dirs(extracted_dir)
        assert len(node_dirs) == 1
        assert node_dirs[0].name == "test_mock_node"
        extracted_dir_2 = download_and_extract_onnx_tests(base_dir)
        assert extracted_dir_2 == extracted_dir


def test_downloader_missing_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        def mock_retrieve(url, filename):
            mem_zip = io.BytesIO()
            with zipfile.ZipFile(mem_zip, mode="w") as zf:
                z_info = zipfile.ZipInfo("wrong_path/")
                zf.writestr(z_info, b"")
            with open(filename, "wb") as f:
                f.write(mem_zip.getvalue())

        monkeypatch.setattr(urllib.request, "urlretrieve", mock_retrieve)
        with pytest.raises(FileNotFoundError):
            download_and_extract_onnx_tests(base_dir)
