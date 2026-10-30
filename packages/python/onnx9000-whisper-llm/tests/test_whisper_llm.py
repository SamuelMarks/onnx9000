"""Comprehensive unit tests for onnx9000.whisper_llm and onnx9000_whisper_llm."""

import onnx9000_whisper_llm
import pytest
from onnx9000.whisper_llm import Whisper, WhisperLLMPipeline, transcribe


def test_transcribe_and_whisper_pipeline() -> None:
    """Verify transcribe() and WhisperLLMPipeline audio buffering and model instantiation."""
    with pytest.raises(ValueError, match="Invalid audio string"):
        transcribe("")
    assert transcribe("test") == "[Whisper-LLM] transcribed test"
    assert onnx9000_whisper_llm.transcribe("clip.wav") == "[Whisper-LLM] transcribed clip.wav"

    pipe = WhisperLLMPipeline()
    assert isinstance(pipe.model, Whisper)
    assert pipe.push_audio([0.1, -0.1, 0.2]) == 3
    assert pipe.transcribe("hello") == "[Whisper-LLM] transcribed hello"
