"""Provide logging functionality for GenAI generation stats."""

import logging

logger = logging.getLogger(__name__)


class GenerationStatsLogger:
    """Logs generation statistics."""

    def __init__(self) -> None:
        """Initialize GenerationStatsLogger with configured parameters and state."""
        self.stats: dict[str, int | float | str | bool | bytes | list | dict | tuple | None] = {}

    def record(
        self, key: str, value: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> None:
        """Record a statistic."""
        self.stats[key] = value

    def log(self) -> bool:
        """Log current statistics."""
        logger.info(f"Generation Stats: {self.stats}")
        return True
