import math

from onnx9000.diffusers.schedulers import DDIMScheduler, EulerDiscreteScheduler


def test_ddim_scheduler_step():
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    sample = [1.0]
    model_output = [0.1]
    timestep = 999

    out = scheduler.step(model_output, timestep, sample)
    assert len(out) == 1
    assert isinstance(out[0], float)


def test_euler_scheduler_step():
    scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)
    sample = [1.0]
    model_output = [0.1]
    timestep = 500

    out = scheduler.step(model_output, timestep, sample)
    assert len(out) == 1
    assert math.isclose(out[0], 1.05, rel_tol=1e-5)
