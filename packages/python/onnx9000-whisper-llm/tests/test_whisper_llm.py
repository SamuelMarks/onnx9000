import pytest
from onnx9000_whisper_llm import transcribe


def test_transcribe():
    assert transcribe("audio") == "[Whisper-LLM] transcribed audio"


def test_transcribe_invalid():
    with pytest.raises(ValueError):
        transcribe("")
