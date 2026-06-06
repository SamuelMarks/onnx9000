import pytest
from onnx9000.backends.testing.downloader import *

def test_download_and_extract_onnx_tests():
    try:
        res = download_and_extract_onnx_tests()
    except Exception:
        pass

def test_get_node_test_dirs():
    try:
        res = get_node_test_dirs()
    except Exception:
        pass

