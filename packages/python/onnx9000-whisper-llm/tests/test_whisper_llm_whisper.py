"""Unit tests and verification suite for whisper functionality."""

import pytest
from onnx9000_whisper_llm import transcribe


def test_transcribe() -> None:
    """Verify that transcribe behaves as expected."""
    with pytest.raises(ValueError):
        transcribe("")
    assert transcribe("test") == "[Whisper-LLM] transcribed test"
