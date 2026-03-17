"""Tests the logger module functionality."""

import logging
import sys
from unittest.mock import patch

from onnx9000.core import config
from onnx9000.core.logger import get_logger, root_logger


def test_get_logger_creation() -> None:
    """Tests the test_get_logger_creation functionality."""
    test_logger = get_logger("test_new_logger")
    assert test_logger.name == "test_new_logger"
    assert len(test_logger.handlers) == 1
    assert isinstance(test_logger.handlers[0], logging.StreamHandler)
    assert test_logger.handlers[0].stream == sys.stdout


def test_get_logger_existing() -> None:
    """Tests the test_get_logger_existing functionality."""
    get_logger("test_existing_logger")
    test_logger2 = get_logger("test_existing_logger")
    assert len(test_logger2.handlers) == 1


def test_get_logger_level_config() -> None:
    """Tests the test_get_logger_level_config functionality."""
    with patch.object(config, "LOG_LEVEL", "DEBUG", create=True):
        test_logger = get_logger("test_debug_logger")
        assert test_logger.level == logging.DEBUG


def test_get_logger_invalid_level_config() -> None:
    """Tests the test_get_logger_invalid_level_config functionality."""
    with patch.object(config, "LOG_LEVEL", "INVALID_LEVEL", create=True):
        test_logger = get_logger("test_invalid_logger")
        assert test_logger.level == logging.INFO


def test_root_logger() -> None:
    """Tests the test_root_logger functionality."""
    assert root_logger.name == "onnx9000"
