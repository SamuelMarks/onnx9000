import asyncio
import math
import os
from unittest.mock import MagicMock, patch

import pytest
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
    ProgressBarConfig,
    PyTorchPCG,
    fetch_hub_file,
    parse_model_index,
    rand,
    randn,
    set_progress_bar_config,
)


# --- pipeline.py ---
def test_diffusion_pipeline():
    pipe = DiffusionPipeline.from_pretrained("test_path", foo="bar")
    assert pipe.config["model_path"] == "test_path"
    assert pipe.config["foo"] == "bar"

    cb_calls = []

    def cb(step, t, total, latents):
        cb_calls.append(step)

    async def run_pipe():
        return await pipe("prompt", callback_on_step_end=cb, num_inference_steps=2)

    res = asyncio.run(run_pipe())
    assert "images" in res
    assert len(cb_calls) == 2

    def cb_abort(step, t, total, latents):
        pipe.free_memory()

    async def run_pipe_abort():
        return await pipe("prompt", callback_on_step_end=cb_abort, num_inference_steps=5)

    asyncio.run(run_pipe_abort())

    pb_config1()


# --- registry.py ---
def test_registry():
    @register_op("my_domain", "my_op")
    class MyOp:
        pass

    assert MyOp._domain == "my_domain"
    assert MyOp._op_name == "my_op"


# --- schedulers.py ---
def test_schedulers():
    s = Scheduler(10)
    s.set_timesteps(5, "leading")
    s.set_timesteps(5, "trailing")
    s.set_timesteps(5, "linspace")
    with pytest.raises(ValueError):
        s.set_timesteps(5, "invalid")

    assert s.scale_model_input([1.0], 0) == [1.0]
    assert s.step([1.0], 0, [1.0]) is None

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
def test_utils():
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


@patch("onnx9000_diffusers.utils.urllib.request.urlopen")
def test_fetch_hub_file(mock_urlopen, tmp_path):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response

    def copyfileobj_side_effect(src, dst):
        dst.write(b"test data")

    with patch(
        "onnx9000_diffusers.utils.shutil.copyfileobj",
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
        def copy_fail(src, dst):
            dst.write(b"test data")
            raise Exception("failed")

        with patch("onnx9000_diffusers.utils.shutil.copyfileobj", side_effect=copy_fail):
            with pytest.raises(Exception):
                fetch_hub_file("fail/repo", "fail.json", cache_dir)


@patch("onnx9000_diffusers.utils.fetch_hub_file")
def test_parse_model_index(mock_fetch, tmp_path):
    cache_dir = str(tmp_path / "cache")
    file_path = os.path.join(cache_dir, "test.json")
    os.makedirs(cache_dir, exist_ok=True)
    with open(file_path, "w") as f:
        f.write('{"key": "value"}')

    mock_fetch.return_value = file_path

    res = parse_model_index("test/repo", cache_dir)
    assert res == {"key": "value"}
