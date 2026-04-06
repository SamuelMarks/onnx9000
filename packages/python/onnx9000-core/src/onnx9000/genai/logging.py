"""Provide logging functionality for GenAI generation stats."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GenerationStatsLogger:
    """Logs generation statistics."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.stats: dict[str, Any] = {}

    def record(self, key: str, value: Any) -> None:
        """Record a statistic."""
        self.stats[key] = value

    def log(self) -> bool:
        """Log current statistics."""
        logger.info(f"Generation Stats: {self.stats}")
        return True
