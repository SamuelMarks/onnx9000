"""Unit tests for GenAI audio synthesis and streaming in `onnx9000.genai.audio`."""

import pytest
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


def test_audio_models_and_streaming() -> None:
    """Verify VITSModel, BarkModel, MusicGenModel, StreamingAudioOutput, MelSpectrogramLoop, WebAudioAPIIntegrator, VocoderDecoder, MultiSpeakerEmbeddings, ContinuousAudioGenerator, and WavExporter."""
    vits = VITSModel({"rate": 22050})
    with pytest.raises(RuntimeError):
        vits.synthesize("hi")
    vits.load()
    assert len(vits.synthesize("hi")) == 6

    bark = BarkModel()
    bark.set_history_prompt("v2/en_speaker_0")
    assert bark.history_prompt == "v2/en_speaker_0"
    assert len(bark.generate("hello")) == 10

    mg = MusicGenModel()
    assert len(mg.generate_music("jazz", 1)) == 32000

    stream = StreamingAudioOutput()
    stream.push([0.5, 0.25])
    assert stream.pop(1) == [0.5]
    assert stream.buffer == [0.25]

    mel = MelSpectrogramLoop()
    mel.add_spectrogram("s0")
    assert mel.process_all() == 1

    web = WebAudioAPIIntegrator()
    web.create_context()
    assert web.play([0.1, 0.2]) is True

    voc = VocoderDecoder()
    assert voc.decode([1.0, 2.0]) == [2.0, 4.0]

    spk = MultiSpeakerEmbeddings()
    spk.register_speaker("alice", [0.1, 0.9])
    assert spk.get_speaker("alice") == [0.1, 0.9]

    gen = ContinuousAudioGenerator()
    gen.start()
    assert gen.is_running is True
    gen.stop()
    assert gen.is_running is False

    wav = WavExporter()
    assert wav.export([0.1, -0.1], "out.wav") is True
