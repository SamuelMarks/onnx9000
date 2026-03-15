"""
Custom logging hierarchy for onnx9000 replacing raw print statements.
"""

import logging
import sys

from onnx9000.core import config


def get_logger(name: str) -> logging.Logger:
    """
    Get a custom logger for onnx9000 with the specified name.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(
        getattr(logging, getattr(config, "LOG_LEVEL", "INFO").upper(), logging.INFO)
    )
    return logger


# Root logger for the package
root_logger = get_logger("onnx9000")
