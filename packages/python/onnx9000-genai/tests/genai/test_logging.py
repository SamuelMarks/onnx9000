"""Unit tests and verification suite for logging functionality."""

from onnx9000.genai.logging import GenerationStatsLogger


def test_stats_logger() -> None:
    """Verify that stats logger behaves as expected."""
    logger = GenerationStatsLogger()
    logger.record("key1", 100)
    assert logger.stats["key1"] == 100
    assert logger.log()
