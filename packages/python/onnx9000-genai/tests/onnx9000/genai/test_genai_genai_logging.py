"""Unit tests for GenerationStatsLogger in `onnx9000.genai.logging`."""

from onnx9000.genai.logging import GenerationStatsLogger


def test_generation_stats_logger() -> None:
    """Verify GenerationStatsLogger records and logs metrics."""
    logger = GenerationStatsLogger()
    logger.record("tps", 42.5)
    assert logger.stats["tps"] == 42.5
    assert logger.log() is True
