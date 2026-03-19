def test_audio():
    from onnx9000.genai.audio import (
        VITSModel,
        BarkModel,
        MusicGenModel,
        StreamingAudioOutput,
        MelSpectrogramLoop,
        WebAudioAPIIntegrator,
        VocoderDecoder,
        MultiSpeakerEmbeddings,
        ContinuousAudioGenerator,
        WavExporter,
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
