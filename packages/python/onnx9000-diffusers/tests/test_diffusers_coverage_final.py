"""Comprehensive tests for onnx9000-diffusers to reach 100% coverage."""

import asyncio

import pytest
from onnx9000.diffusers.models import AutoencoderKL, UNet2DConditionModel
from onnx9000.diffusers.pipeline import DiffusionPipeline, set_progress_bar_config
from onnx9000.diffusers.registry import register_op
from onnx9000.diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
)


def test_autoencoder_kl() -> None:
    """Verify AutoencoderKL encode/decode logic."""
    vae = AutoencoderKL()
    data = [1.0, 2.0, 3.0]
    latents = vae.encode(data)
    decoded = vae.decode(latents)
    assert len(latents) == 3
    assert len(decoded) == 3
    assert pytest.approx(decoded[0]) == 1.0


def test_unet_model() -> None:
    """Verify UNet2DConditionModel denoising step."""
    unet = UNet2DConditionModel()
    sample = [1.0, 1.0]
    # sample - (timestep * 0.01)
    res = unet(sample, timestep=10, encoder_hidden_states=[])
    assert res == [0.9, 0.9]


def test_diffusion_pipeline_sync_wrapper() -> None:
    """Verify DiffusionPipeline execution and abortion using asyncio.run."""

    async def run_test() -> None:
        """Test helper run_test."""
        pipe = DiffusionPipeline.from_pretrained("dummy_path", custom_attr=True)
        assert pipe.config["model_path"] == "dummy_path"
        assert pipe.config["custom_attr"] is True

        steps_completed = 0

        def callback(
            step: int | float | str | bool | bytes | list | dict | tuple | None,
            timestep: int | float | str | bool | bytes | list | dict | tuple | None,
            total_steps: int | float | str | bool | bytes | list | dict | tuple | None,
            latents: int | float | str | bool | bytes | list | dict | tuple | None,
        ) -> None:
            """Callback."""
            nonlocal steps_completed
            steps_completed += 1
            if step == 2:
                pipe.free_memory()  # Should set _is_aborted to True

        res = await pipe(
            prompt="test prompt", callback_on_step_end=callback, num_inference_steps=10
        )
        # Should have stopped after step 2 (0, 1, 2 = 3 steps)
        assert steps_completed == 3
        assert "images" in res

    asyncio.run(run_test())


def test_progress_bar() -> None:
    """Verify no-op progress bar config."""
    assert set_progress_bar_config(enabled=True) is None


def test_registry_decorator() -> None:
    """Verify register_op decorator correctly sets metadata."""

    @register_op("ai.onnx", "TestOp")
    class TestOp:
        """Test fixture TestOp."""

        registered = True

    assert TestOp._domain == "ai.onnx"
    assert TestOp._op_name == "TestOp"
    assert TestOp.registered is True


def test_schedulers_gaps() -> None:
    """Verify remaining branches in schedulers."""
    # DDIMScheduler
    ddim = DDIMScheduler(num_train_timesteps=1000)
    ddim.set_timesteps(10)
    res = ddim.step([1.0], 500, [0.1])
    assert len(res) == 1

    # DDPMScheduler
    ddpm = DDPMScheduler(num_train_timesteps=1000)
    ddpm.set_timesteps(10)
    res = ddpm.step([1.0], 500, [0.1])
    assert len(res) == 1

    # EulerDiscreteScheduler
    euler = EulerDiscreteScheduler(num_train_timesteps=1000)
    euler.set_timesteps(10)
    res = euler.step([1.0], 500, [0.1])
    assert len(res) == 1

    # LCMScheduler
    lcm = LCMScheduler(num_train_timesteps=1000)
    lcm.set_timesteps(10)
    res = lcm.step([1.0], 500, [0.1])
    assert len(res) == 1
