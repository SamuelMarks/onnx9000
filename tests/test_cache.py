"""Module providing core logic and structural definitions."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from onnx9000.utils.cache import clear_cache
from onnx9000.core import config


def test_clear_cache_exists_success(tmp_path):
    """Provides semantic functionality and verification."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with patch.object(config, "ONNX9000_CACHE_DIR", cache_dir):
        clear_cache()
    assert not cache_dir.exists()


def test_clear_cache_not_exists(tmp_path):
    """Provides semantic functionality and verification."""
    cache_dir = tmp_path / "cache"
    with patch.object(config, "ONNX9000_CACHE_DIR", cache_dir):
        clear_cache()
    assert not cache_dir.exists()


def test_clear_cache_error(tmp_path):
    """Provides semantic functionality and verification."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with patch.object(config, "ONNX9000_CACHE_DIR", cache_dir):
        with patch("shutil.rmtree", side_effect=Exception("mocked error")):
            clear_cache()
    assert cache_dir.exists()
