"""Module providing core logic and structural definitions."""

import logging
import sys


def get_logger(name: str = "onnx9000") -> logging.Logger:
    """Return a configured logger for the ONNX9000 package."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger


logger = get_logger()
