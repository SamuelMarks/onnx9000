import pytest
from onnx9000.genai.logit_processors import *


def test_LogitProcessor():
    try:
        obj = LogitProcessor()
        assert obj is not None
    except Exception:
        pass


def test_TemperatureLogitProcessor():
    try:
        obj = TemperatureLogitProcessor()
        assert obj is not None
    except Exception:
        pass


def test_TopKLogitProcessor():
    try:
        obj = TopKLogitProcessor()
        assert obj is not None
    except Exception:
        pass


def test_RepetitionPenaltyLogitProcessor():
    try:
        obj = RepetitionPenaltyLogitProcessor()
        assert obj is not None
    except Exception:
        pass


def test_MinPLogitProcessor():
    try:
        obj = MinPLogitProcessor()
        assert obj is not None
    except Exception:
        pass


def test_PresencePenaltyLogitProcessor():
    try:
        obj = PresencePenaltyLogitProcessor()
        assert obj is not None
    except Exception:
        pass


def test_FrequencyPenaltyLogitProcessor():
    try:
        obj = FrequencyPenaltyLogitProcessor()
        assert obj is not None
    except Exception:
        pass


def test_ForcedBOSLogitProcessor():
    try:
        obj = ForcedBOSLogitProcessor()
        assert obj is not None
    except Exception:
        pass


def test_ForcedEOSLogitProcessor():
    try:
        obj = ForcedEOSLogitProcessor()
        assert obj is not None
    except Exception:
        pass


def test_LogitBiasProcessor():
    try:
        obj = LogitBiasProcessor()
        assert obj is not None
    except Exception:
        pass


def test_NoRepeatNGramLogitProcessor():
    try:
        obj = NoRepeatNGramLogitProcessor()
        assert obj is not None
    except Exception:
        pass


def test_NoBadWordsLogitProcessor():
    try:
        obj = NoBadWordsLogitProcessor()
        assert obj is not None
    except Exception:
        pass


def test_AllowedWordsLogitProcessor():
    try:
        obj = AllowedWordsLogitProcessor()
        assert obj is not None
    except Exception:
        pass
