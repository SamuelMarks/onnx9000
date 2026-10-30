"""Unit tests and verification suite for diffusers all functionality."""

import asyncio
import math
import os
from typing import NoReturn
from unittest.mock import MagicMock, patch

import pytest
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.diffusers.pipeline import DiffusionPipeline
from onnx9000.diffusers.pipeline import set_progress_bar_config as pb_config1
from onnx9000.diffusers.registry import register_op
from onnx9000_diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DDPMWuerstchenScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    SASolverScheduler,
    Scheduler,
    UniPCMultistepScheduler,
)
from onnx9000_diffusers.utils import (
    PyTorchPCG,
    fetch_hub_file,
    parse_model_index,
    rand,
    randn,
    set_progress_bar_config,
)


# --- pipeline.py ---
def test_diffusion_pipeline() -> None:
    """Verify that diffusion pipeline behaves as expected.

    Returns:
        Result of test diffusion pipeline.
    """
    pipe = DiffusionPipeline.from_pretrained("test_path", foo="bar")
    assert pipe.config["model_path"] == "test_path"
    assert pipe.config["foo"] == "bar"

    cb_calls = []

    def cb(
        step: int | float | str | bool | bytes | list | dict | tuple | None,
        t: int | float | str | bool | bytes | list | dict | tuple | None,
        total: int | float | str | bool | bytes | list | dict | tuple | None,
        latents: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> None:
        """Execute cb operation.

        Args:
            step: Parameter specifying step.
            t: Parameter specifying t.
            total: Parameter specifying total.
            latents: Parameter specifying latents.
        """
        cb_calls.append(step)

    async def run_pipe() -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute run pipe operation.

        Returns:
            Result of run pipe.
        """
        return await pipe("prompt", callback_on_step_end=cb, num_inference_steps=2)

    res = asyncio.run(run_pipe())
    assert "images" in res
    assert len(cb_calls) == 2

    async def run_pipe_no_cb() -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute run pipe no cb operation.

        Returns:
            Result of run pipe no cb.
        """
        return await pipe("prompt", callback_on_step_end=None, num_inference_steps=1)

    res_no_cb = asyncio.run(run_pipe_no_cb())
    assert "images" in res_no_cb

    def cb_abort(
        step: int | float | str | bool | bytes | list | dict | tuple | None,
        t: int | float | str | bool | bytes | list | dict | tuple | None,
        total: int | float | str | bool | bytes | list | dict | tuple | None,
        latents: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> None:
        """Execute cb abort operation.

        Args:
            step: Parameter specifying step.
            t: Parameter specifying t.
            total: Parameter specifying total.
            latents: Parameter specifying latents.
        """
        pipe.free_memory()

    async def run_pipe_abort() -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute run pipe abort operation.

        Returns:
            Result of run pipe abort.
        """
        return await pipe("prompt", callback_on_step_end=cb_abort, num_inference_steps=5)

    asyncio.run(run_pipe_abort())

    pb_config1()


# --- registry.py ---
def test_registry() -> None:
    """Verify that registry behaves as expected."""

    @register_op("my_domain", "my_op")
    class MyOp:
        """Represents my op."""

        pass

    assert MyOp._domain == "my_domain"
    assert MyOp._op_name == "my_op"


# --- schedulers.py ---
def test_schedulers() -> None:
    """Verify that schedulers behaves as expected."""
    s = Scheduler(10)
    s.set_timesteps(5, "leading")
    s.set_timesteps(5, "trailing")
    s.set_timesteps(5, "linspace")
    with pytest.raises(ValueError):
        s.set_timesteps(5, "invalid")

    assert s.scale_model_input([1.0], 0) == [1.0]
    assert s.step([1.0], 0, [1.0]) == [1.0]

    s.alphas_cumprod = [0.9] * 10
    res = s.add_noise([1.0], [0.1], 0)
    assert len(res) == 1

    with pytest.raises(ValueError):
        s.alphas_cumprod = [math.nan] * 10
        s.add_noise([1.0], [0.1], 0)

    ddpm = DDPMScheduler(10)
    ddpm.set_timesteps(5)
    ddpm.step([0.1], 2, [1.0])

    ddim = DDIMScheduler(10)
    ddim.set_timesteps(5)
    ddim.step([0.1], 2, [1.0])

    euler = EulerDiscreteScheduler(10)
    euler.set_timesteps(5)
    euler.scale_model_input([1.0], 2)
    euler.step([0.1], 2, [1.0])

    euler_karras = EulerDiscreteScheduler(10, use_karras_sigmas=True)
    euler_karras.set_timesteps(5)

    lcm = LCMScheduler(10)
    lcm.step([0.1], 0, [1.0])

    assert DDPMWuerstchenScheduler.__dummy__

    flow = FlowMatchEulerDiscreteScheduler(10)
    flow.set_timesteps(5)
    flow.step([0.1], 0, [1.0])

    sa = SASolverScheduler(10)
    sa.step([0.1], 0, [1.0])

    euler_a = EulerAncestralDiscreteScheduler(10)
    euler_a.set_timesteps(5)
    euler_a.step([0.1], euler_a.timesteps[0], [1.0], generator=True)
    euler_a.step([0.1], euler_a.timesteps[0], [1.0], generator=None)

    PNDMScheduler(10)
    LMSDiscreteScheduler(10)

    dpm_multi = DPMSolverMultistepScheduler(10)
    dpm_multi.step([0.1], 0, [1.0])

    dpm_single = DPMSolverSinglestepScheduler(10)
    dpm_single.step([0.1], 0, [1.0])

    KDPM2DiscreteScheduler(10)
    KDPM2AncestralDiscreteScheduler(10)
    HeunDiscreteScheduler(10)

    unipc = UniPCMultistepScheduler(10)
    unipc.step([0.1], 0, [1.0])


# --- utils.py ---
def test_utils() -> None:
    """Verify that utils behaves as expected."""
    pcg = PyTorchPCG(42)
    val1 = pcg.next_float()
    val2 = pcg.next_uint()
    assert isinstance(val1, float)
    assert isinstance(val2, int)

    t1 = rand((2, 2), pcg)
    assert len(t1) == 4

    t2 = randn((2, 2), pcg)
    assert len(t2) == 4

    set_progress_bar_config(False)


@patch("onnx9000.diffusers.utils.urllib.request.urlopen")
def test_fetch_hub_file(
    mock_urlopen: int | float | str | bool | bytes | list | dict | tuple | None,
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    # Setup mock response
    """Verify that fetch hub file behaves as expected.

    Args:
        mock_urlopen: Parameter specifying mock urlopen.
        tmp_path: Parameter specifying tmp path.

    Raises:
        Exception: If an invalid state or argument is encountered during execution.
    """
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response

    def copyfileobj_side_effect(
        src: int | float | str | bool | bytes | list | dict | tuple | None,
        dst: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> None:
        """Execute copyfileobj side effect operation.

        Args:
            src: Parameter specifying src.
            dst: Parameter specifying dst.
        """
        dst.write(b"test data")

    with patch(
        "onnx9000.diffusers.utils.shutil.copyfileobj",
        side_effect=copyfileobj_side_effect,
    ):
        mock_urlopen.return_value = mock_response

        cache_dir = str(tmp_path / "cache")
        path = fetch_hub_file("test/repo", "model_index.json", cache_dir)
        assert os.path.exists(path)

        # test cache
        path2 = fetch_hub_file("test/repo", "model_index.json", cache_dir)
        assert path == path2

        # test fetch failure with copyfileobj throwing
        def copy_fail(
            src: int | float | str | bool | bytes | list | dict | tuple | None,
            dst: int | float | str | bool | bytes | list | dict | tuple | None,
        ) -> NoReturn:
            """Execute copy fail operation.

            Args:
                src: Parameter specifying src.
                dst: Parameter specifying dst.

            Raises:
                Exception: If an invalid state or argument is encountered during execution.
            """
            dst.write(b"test data")
            raise Onnx9000Error("failed")

        with patch("onnx9000.diffusers.utils.shutil.copyfileobj", side_effect=copy_fail):
            with pytest.raises(
                (Onnx9000Error, ValueError, RuntimeError, TypeError, KeyError, AttributeError)
            ):
                fetch_hub_file("fail/repo", "fail.json", cache_dir)


@patch("onnx9000.diffusers.utils.fetch_hub_file")
def test_parse_model_index(
    mock_fetch: int | float | str | bool | bytes | list | dict | tuple | None,
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that parse model index behaves as expected.

    Args:
        mock_fetch: Parameter specifying mock fetch.
        tmp_path: Parameter specifying tmp path.
    """
    cache_dir = str(tmp_path / "cache")
    file_path = os.path.join(cache_dir, "test.json")
    os.makedirs(cache_dir, exist_ok=True)
    with open(file_path, "w") as f:
        f.write('{"key": "value"}')

    mock_fetch.return_value = file_path

    res = parse_model_index("test/repo", cache_dir)
    assert res == {"key": "value"}
