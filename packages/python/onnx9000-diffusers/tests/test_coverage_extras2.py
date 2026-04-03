import math
from unittest.mock import patch

import numpy as np
import pytest
from onnx9000_diffusers.schedulers import EulerDiscreteScheduler
from onnx9000_diffusers.utils import fetch_hub_file, set_progress_bar_config


def test_euler_scale():
    euler = EulerDiscreteScheduler()
    euler.set_timesteps(10)
    assert len(euler.scale_model_input([1.0], euler.timesteps[0])) == 1
    assert len(euler.scale_model_input([1.0], 9999)) == 1


def test_utils_progress():
    set_progress_bar_config(False)
    from onnx9000_diffusers.utils import global_progress_bar_config

    assert not global_progress_bar_config.enabled


def test_utils_fetch_error(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()

    def mock_urlopen(*args, **kwargs):
        # Create a dummy file before crashing to test os.remove
        import os

        with open(os.path.join(cache, "testfile"), "w") as f:
            f.write("test")
        raise Exception("Mock Error")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        with pytest.raises(Exception):
            fetch_hub_file("repo/id", "testfile", str(cache))
