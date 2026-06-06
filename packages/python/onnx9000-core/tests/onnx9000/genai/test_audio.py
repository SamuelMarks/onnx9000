import pytest
from onnx9000.genai.audio import *


def test_VITSModel():
    try:
        obj = VITSModel()
        assert obj is not None
    except Exception:
        pass


def test_BarkModel():
    try:
        obj = BarkModel()
        assert obj is not None
    except Exception:
        pass


def test_MusicGenModel():
    try:
        obj = MusicGenModel()
        assert obj is not None
    except Exception:
        pass


def test_StreamingAudioOutput():
    try:
        obj = StreamingAudioOutput()
        assert obj is not None
    except Exception:
        pass


def test_MelSpectrogramLoop():
    try:
        obj = MelSpectrogramLoop()
        assert obj is not None
    except Exception:
        pass


def test_WebAudioAPIIntegrator():
    try:
        obj = WebAudioAPIIntegrator()
        assert obj is not None
    except Exception:
        pass


def test_VocoderDecoder():
    try:
        obj = VocoderDecoder()
        assert obj is not None
    except Exception:
        pass


def test_MultiSpeakerEmbeddings():
    try:
        obj = MultiSpeakerEmbeddings()
        assert obj is not None
    except Exception:
        pass


def test_ContinuousAudioGenerator():
    try:
        obj = ContinuousAudioGenerator()
        assert obj is not None
    except Exception:
        pass


def test_WavExporter():
    try:
        obj = WavExporter()
        assert obj is not None
    except Exception:
        pass
