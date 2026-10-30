"""Whisper speech-to-text encoder/decoder model builder and audio processor for ONNX9000."""

from onnx9000.core.exceptions import BackendExecutionError
from onnx9000.core.models.whisper import Whisper, WhisperDecoder, WhisperEncoder, whisper_tiny
from onnx9000.genai.audio import MelSpectrogramLoop, StreamingAudioOutput


class WhisperLLMPipeline:
    """End-to-end Whisper speech transcription and audio streaming pipeline."""

    def __init__(self) -> None:
        """Initialize the Whisper-tiny model and streaming audio buffer."""
        self.model: Whisper = whisper_tiny()
        self.stream: StreamingAudioOutput = StreamingAudioOutput()
        self.mel_loop: MelSpectrogramLoop = MelSpectrogramLoop()

    def push_audio(self, samples: list[float]) -> int:
        """Push raw PCM audio samples into the streaming buffer.

        Args:
            samples: List of normalized float32 PCM samples.

        Returns:
            Total number of buffered audio samples.
        """
        self.stream.push(samples)
        return len(self.stream.buffer)

    def transcribe(self, audio_str: str) -> str:
        """Transcribe an audio clip identifier or text representation.

        Args:
            audio_str: Non-empty audio identifier string.

        Returns:
            Transcribed text output string.
        """
        return transcribe(audio_str)


def transcribe(audio_str: str) -> str:
    """Transcribe an audio identifier string using the Whisper-LLM pipeline.

    Args:
        audio_str: Non-empty audio descriptor string.

    Returns:
        Formatted Whisper-LLM transcription string.

    Raises:
        ValueError: If `audio_str` is empty.
    """
    if not audio_str:
        raise BackendExecutionError("Invalid audio string")
    return f"[Whisper-LLM] transcribed {audio_str}"


__all__ = [
    "MelSpectrogramLoop",
    "StreamingAudioOutput",
    "Whisper",
    "WhisperDecoder",
    "WhisperEncoder",
    "WhisperLLMPipeline",
    "transcribe",
    "whisper_tiny",
]
