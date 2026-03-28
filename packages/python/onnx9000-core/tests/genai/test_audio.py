"""Module providing functionality for test_audio."""

"""Test audio."""


def test_audio():
    """Docstring."""
    from onnx9000.genai.audio import (
        BarkModel,
        ContinuousAudioGenerator,
        MelSpectrogramLoop,
        MultiSpeakerEmbeddings,
        MusicGenModel,
        StreamingAudioOutput,
        VITSModel,
        VocoderDecoder,
        WavExporter,
        WebAudioAPIIntegrator,
    )

    assert VITSModel()._initialized
    assert BarkModel()._initialized
    assert MusicGenModel()._initialized
    assert StreamingAudioOutput()._initialized
    assert MelSpectrogramLoop()._initialized
    assert WebAudioAPIIntegrator()._initialized
    assert VocoderDecoder()._initialized
    assert MultiSpeakerEmbeddings()._initialized
    assert ContinuousAudioGenerator()._initialized
    assert WavExporter()._initialized
