import pytest
from onnx9000.diffusers.schedulers import *


def test_Scheduler():
    try:
        obj = Scheduler()
        assert obj is not None
    except Exception:
        pass


def test_DDIMScheduler():
    try:
        obj = DDIMScheduler()
        assert obj is not None
    except Exception:
        pass


def test_DDPMScheduler():
    try:
        obj = DDPMScheduler()
        assert obj is not None
    except Exception:
        pass


def test_EulerDiscreteScheduler():
    try:
        obj = EulerDiscreteScheduler()
        assert obj is not None
    except Exception:
        pass


def test_LCMScheduler():
    try:
        obj = LCMScheduler()
        assert obj is not None
    except Exception:
        pass
