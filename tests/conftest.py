"""Module docstring."""

import shutil
import tempfile
from pathlib import Path

import pytest

from onnx9000 import config


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Sets up a clean testing environment."""
    # Use a temporary directory for JIT compilation tests to avoid polluting user cache.
    test_cache = Path(tempfile.mkdtemp(prefix="onnx9000_test_cache_"))
    config.ONNX9000_CACHE_DIR = test_cache

    yield

    # Teardown: clean up the test cache directory.
    if test_cache.exists():
        shutil.rmtree(test_cache)


@pytest.fixture
def temp_dir():
    """Provides a temporary directory for a specific test."""
    path = Path(tempfile.mkdtemp(prefix="onnx9000_test_"))
    yield path
    shutil.rmtree(path)
