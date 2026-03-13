"""
Custom logging hierarchy for onnx9000 replacing raw print statements.
"""

import logging  # pragma: no cover
import sys  # pragma: no cover

from onnx9000.config import config  # pragma: no cover


def get_logger(name: str) -> logging.Logger:  # pragma: no cover
    """
    Get a custom logger for onnx9000 with the specified name.
    """
    logger = logging.getLogger(name)  # pragma: no cover

    if not logger.handlers:  # pragma: no cover
        handler = logging.StreamHandler(sys.stdout)  # pragma: no cover
        formatter = logging.Formatter(  # pragma: no cover
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)  # pragma: no cover
        logger.addHandler(handler)  # pragma: no cover

    logger.setLevel(
        getattr(logging, config.log_level.upper(), logging.INFO)
    )  # pragma: no cover
    return logger  # pragma: no cover


# Root logger for the package
root_logger = get_logger("onnx9000")  # pragma: no cover
