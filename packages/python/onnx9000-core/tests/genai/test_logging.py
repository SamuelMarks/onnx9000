import pytest
from onnx9000.genai.logging import GenerationStatsLogger


def test_stats_logger():
    logger = GenerationStatsLogger()
    logger.record("key1", 100)
    assert logger.stats["key1"] == 100
    assert logger.log()
