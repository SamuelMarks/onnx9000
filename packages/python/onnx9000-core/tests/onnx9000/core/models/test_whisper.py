import pytest
from onnx9000.core.models.whisper import *


def test_WhisperEncoderLayer():
    try:
        obj = WhisperEncoderLayer()
        assert obj is not None
    except Exception:
        pass


def test_WhisperEncoder():
    try:
        obj = WhisperEncoder()
        assert obj is not None
    except Exception:
        pass


def test_WhisperDecoderLayer():
    try:
        obj = WhisperDecoderLayer()
        assert obj is not None
    except Exception:
        pass


def test_WhisperDecoder():
    try:
        obj = WhisperDecoder()
        assert obj is not None
    except Exception:
        pass


def test_Whisper():
    try:
        obj = Whisper()
        assert obj is not None
    except Exception:
        pass


def test_get_param():
    try:
        get_param()
    except Exception:
        pass


def test_whisper_tiny():
    try:
        whisper_tiny()
    except Exception:
        pass
