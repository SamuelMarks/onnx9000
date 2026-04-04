"""Module docstring."""

import math

from onnx9000_diffusers.schedulers import (
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    LCMScheduler,
    Scheduler,
)


def test_scheduler_spacing():
    """Docstring for D103."""
    scheduler = Scheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(10, spacing="leading")
    assert scheduler.timesteps[0] == 900

    scheduler.set_timesteps(10, spacing="trailing")
    assert scheduler.timesteps[-1] == 99

    scheduler.set_timesteps(10, spacing="linspace")
    assert len(scheduler.timesteps) == 10


def test_add_noise():
    """Docstring for D103."""
    scheduler = EulerDiscreteScheduler()
    # Mock betas setup in Euler
    original = [1.0, 0.5]
    noise = [0.1, -0.1]
    noisy = scheduler.add_noise(original, noise, 500)
    assert len(noisy) == 2
    assert not math.isnan(noisy[0])


def test_lcm_scheduler():
    """Docstring for D103."""
    scheduler = LCMScheduler()
    sample = [1.0, -1.0]
    model_output = [0.5, -0.5]
    out = scheduler.step(model_output, 500, sample)
    assert out[0] == 1.0 - 0.05 * 0.5


def test_flow_match_scheduler():
    """Docstring for D103."""
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(50)
    assert len(scheduler.timesteps) == 50
    assert len(scheduler.sigmas) == 51


def test_karras_sigmas():
    """Docstring for D103."""
    scheduler = EulerDiscreteScheduler(use_karras_sigmas=True)
    scheduler.set_timesteps(10)
    assert len(scheduler.sigmas) == 11
    assert scheduler.sigmas[-1] == 0.0
