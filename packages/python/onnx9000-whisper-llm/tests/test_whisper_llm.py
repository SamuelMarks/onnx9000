import pytest
from onnx9000_whisper_llm import transcribe


def test_transcribe():
    with pytest.raises(ValueError):
        transcribe("")
    assert transcribe("test") == "[Whisper-LLM] transcribed test"
