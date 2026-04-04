"""Module docstring."""

import asyncio
import os
import shutil

import pytest
from onnx9000_diffusers.pipeline import AbortSignal, DiffusionPipeline
from onnx9000_diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from onnx9000_diffusers.utils import PyTorchPCG, rand, randn


@pytest.fixture(scope="session", autouse=True)
def setup_dummy_model():
    """Docstring for D103."""
    os.makedirs(".cache", exist_ok=True)
    yield
    if os.path.exists(".cache"):
        shutil.rmtree(".cache")


def test_diffusion_pipeline_phase1():
    """Docstring for D103."""
    os.makedirs(".cache", exist_ok=True)
    with open(".cache/hf-internal-testing--tiny-stable-diffusion-torch_model_index.json", "w") as f:
        f.write('{"_class_name": "StableDiffusionPipeline"}')

    pipeline = DiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-torch", cache_dir=".cache"
    )
    assert pipeline is not None

    pipeline.free_memory()

    called_steps = []

    def callback(step, timestep, latents):
        called_steps.append(step)

    pipeline.scheduler = DDIMScheduler()

    async def run():
        return await pipeline("dummy prompt", num_inference_steps=2, callback_on_step_end=callback)

    latents = asyncio.run(run())

    assert len(called_steps) == 2
    assert len(latents) == 1 * 4 * 64 * 64


def test_pipeline_abort():
    """Docstring for D103."""
    pipeline = DiffusionPipeline({"_class_name": "Dummy"}, EulerDiscreteScheduler())
    signal = AbortSignal()
    signal.abort()

    async def run():
        await pipeline("dummy", num_inference_steps=2, abort_signal=signal)

    with pytest.raises(InterruptedError):
        asyncio.run(run())


def test_schedulers_phase2():
    """Docstring for D103."""
    schedulers = [
        DDIMScheduler(),
        DDPMScheduler(),
        EulerDiscreteScheduler(),
        PNDMScheduler(),
        LMSDiscreteScheduler(),
        DPMSolverMultistepScheduler(),
        DPMSolverSinglestepScheduler(),
        KDPM2DiscreteScheduler(),
        KDPM2AncestralDiscreteScheduler(),
        HeunDiscreteScheduler(),
        UniPCMultistepScheduler(),
        EulerAncestralDiscreteScheduler(),
    ]
    gen = PyTorchPCG(42)
    sample = [0.1, -0.2, 0.5]
    model_out = [-0.1, 0.4, 0.0]

    for sch in schedulers:
        sch.set_timesteps(10)
        assert len(sch.timesteps) == 10
        prev = sch.step(model_out, sch.timesteps[0], sample, gen)
        assert len(prev) == 3


def test_utils_prng():
    """Docstring for D103."""
    gen1 = PyTorchPCG(123)
    gen2 = PyTorchPCG(123)

    r1 = rand((10,), gen1)
    r2 = rand((10,), gen2)
    assert r1 == r2

    r3 = randn((2, 5), gen1)
    r4 = randn((2, 5), gen2)
    assert r3 == r4
