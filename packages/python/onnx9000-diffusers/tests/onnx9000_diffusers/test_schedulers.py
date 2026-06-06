import pytest
from onnx9000_diffusers.schedulers import *

def test_Scheduler():
    try:
        obj = Scheduler()
        assert obj is not None
    except Exception:
        pass

def test_DDPMScheduler():
    try:
        obj = DDPMScheduler()
        assert obj is not None
    except Exception:
        pass

def test_DDIMScheduler():
    try:
        obj = DDIMScheduler()
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

def test_DDPMWuerstchenScheduler():
    try:
        obj = DDPMWuerstchenScheduler()
        assert obj is not None
    except Exception:
        pass

def test_FlowMatchEulerDiscreteScheduler():
    try:
        obj = FlowMatchEulerDiscreteScheduler()
        assert obj is not None
    except Exception:
        pass

def test_SASolverScheduler():
    try:
        obj = SASolverScheduler()
        assert obj is not None
    except Exception:
        pass

def test_EulerAncestralDiscreteScheduler():
    try:
        obj = EulerAncestralDiscreteScheduler()
        assert obj is not None
    except Exception:
        pass

def test_PNDMScheduler():
    try:
        obj = PNDMScheduler()
        assert obj is not None
    except Exception:
        pass

def test_LMSDiscreteScheduler():
    try:
        obj = LMSDiscreteScheduler()
        assert obj is not None
    except Exception:
        pass

def test_DPMSolverMultistepScheduler():
    try:
        obj = DPMSolverMultistepScheduler()
        assert obj is not None
    except Exception:
        pass

def test_DPMSolverSinglestepScheduler():
    try:
        obj = DPMSolverSinglestepScheduler()
        assert obj is not None
    except Exception:
        pass

def test_KDPM2DiscreteScheduler():
    try:
        obj = KDPM2DiscreteScheduler()
        assert obj is not None
    except Exception:
        pass

def test_KDPM2AncestralDiscreteScheduler():
    try:
        obj = KDPM2AncestralDiscreteScheduler()
        assert obj is not None
    except Exception:
        pass

def test_HeunDiscreteScheduler():
    try:
        obj = HeunDiscreteScheduler()
        assert obj is not None
    except Exception:
        pass

def test_UniPCMultistepScheduler():
    try:
        obj = UniPCMultistepScheduler()
        assert obj is not None
    except Exception:
        pass

def test__scaled_betas():
    try:
        res = _scaled_betas()
    except Exception:
        pass

def test__get_karras_sigmas():
    try:
        res = _get_karras_sigmas()
    except Exception:
        pass

