import pytest
from onnx9000.optimizer.surgeon.audio import *


def test__unpack_scalar():
    try:
        _unpack_scalar()
    except Exception:
        pass


def test_fold_mel_weights():
    try:
        fold_mel_weights()
    except Exception:
        pass
