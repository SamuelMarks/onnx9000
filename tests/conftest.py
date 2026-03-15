"""Module providing core logic and structural definitions."""

import shutil
import tempfile
from pathlib import Path
import pytest
from onnx9000.core import config


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Sets up a clean testing environment."""
    test_cache = Path(tempfile.mkdtemp(prefix="onnx9000_test_cache_"))
    config.ONNX9000_CACHE_DIR = test_cache
    yield
    if test_cache.exists():
        shutil.rmtree(test_cache)


@pytest.fixture
def temp_dir():
    """Provides a temporary directory for a specific test."""
    path = Path(tempfile.mkdtemp(prefix="onnx9000_test_"))
    yield path
    shutil.rmtree(path)
