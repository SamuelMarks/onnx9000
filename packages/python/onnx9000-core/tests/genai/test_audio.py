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


def test_vits_model():
    model = VITSModel({"param": 1})
    with pytest.raises(RuntimeError):
        model.synthesize("test")
    model.load()
    out = model.synthesize("te")
    assert len(out) == 6


def test_bark_model():
    model = BarkModel()
    model.set_history_prompt("v2/en_speaker_1")
    assert model.history_prompt == "v2/en_speaker_1"
    assert len(model.generate("test")) == 10


def test_musicgen_model():
    model = MusicGenModel()
    assert len(model.generate_music("pop", 2)) == 64000


def test_streaming_audio():
    stream = StreamingAudioOutput()
    stream.push([0.1, 0.2, 0.3])
    chunk = stream.pop(2)
    assert chunk == [0.1, 0.2]
    assert stream.buffer == [0.3]


def test_mel_spectrogram():
    loop = MelSpectrogramLoop()
    loop.add_spectrogram("spec1")
    assert loop.process_all() == 1


def test_webaudio_api():
    api = WebAudioAPIIntegrator()
    assert not api.play([0.1])
    api.create_context()
    assert api.play([0.1])
    assert not api.play([])


def test_vocoder():
    decoder = VocoderDecoder()
    assert decoder.decode([0.5, 1.0]) == [1.0, 2.0]


def test_speaker_embeddings():
    emb = MultiSpeakerEmbeddings()
    emb.register_speaker("spk1", [0.1, 0.2])
    assert emb.get_speaker("spk1") == [0.1, 0.2]
    assert emb.get_speaker("spk2") is None


def test_continuous_generator():
    gen = ContinuousAudioGenerator()
    gen.start()
    assert gen.is_running
    gen.stop()
    assert not gen.is_running


def test_wav_exporter():
    exporter = WavExporter()
    assert exporter.export([0.1, 0.2], "test.wav")
    assert not exporter.export([], "test.wav")
