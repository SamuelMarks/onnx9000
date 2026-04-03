import math

import numpy as np
import pytest
from onnx9000_diffusers.models import AutoencoderKL
from onnx9000_diffusers.pipeline import DiffusionPipeline
from onnx9000_diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    LCMScheduler,
    SASolverScheduler,
    Scheduler,
)


def test_models_tiling():
    vae = AutoencoderKL("test")
    vae.enable_tiling(128)
    assert getattr(vae, "_tile_size", None) == 128


def test_pipeline_device():
    pipe = DiffusionPipeline({}, None)
    pipe.to("mps")
    assert getattr(pipe, "_device", None) == "mps"
    pipe.free_memory()


def test_scheduler_base():
    sched = Scheduler()
    sched.set_timesteps(10, spacing="linspace")
    assert len(sched.timesteps) == 10

    with pytest.raises(ValueError):
        sched.set_timesteps(10, spacing="unknown")

    sample = [1.0]
    assert sched.scale_model_input(sample, 0) == sample
    assert sched.step([0.0], 0, sample) is None

    with pytest.raises(ValueError):
        sched.alphas_cumprod = [float("nan")]
        sched.add_noise([1.0], [0.0], 0)


def test_other_schedulers():
    lcm = LCMScheduler()
    assert lcm.step([1.0], 0, [2.0]) == [1.95]

    sa = SASolverScheduler()
    assert sa.step([1.0], 0, [2.0]) == [1.98]

    fm = FlowMatchEulerDiscreteScheduler()
    fm.set_timesteps(10)
    assert len(fm.timesteps) == 10
    step_out = fm.step([1.0], fm.timesteps[0], [2.0])
    assert len(step_out) == 1
