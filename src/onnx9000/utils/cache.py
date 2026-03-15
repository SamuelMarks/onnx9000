"""Module providing core logic and structural definitions."""

import shutil
from pathlib import Path

from onnx9000.core import config
from onnx9000.utils.logger import logger


def clear_cache() -> None:
    """
    Clears the entire ONNX9000 JIT compiled extensions cache directory.
    """
    cache_dir: Path = config.ONNX9000_CACHE_DIR
    if cache_dir.exists() and cache_dir.is_dir():
        try:
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared cache directory: {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to clear cache directory: {cache_dir}. Error: {e}")
    else:
        logger.info("Cache directory does not exist. Nothing to clear.")
