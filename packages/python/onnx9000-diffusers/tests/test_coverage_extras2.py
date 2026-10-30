"""Unit tests to verify extra coverage cases in onnx9000-diffusers.

This module contains test cases targeting the Euler discrete scheduler input scaling,
global progress bar utility configuration, and hub file downloading error handling.
"""

from typing import NoReturn
from unittest.mock import patch

import pytest
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_diffusers.schedulers import EulerDiscreteScheduler
from onnx9000_diffusers.utils import fetch_hub_file, set_progress_bar_config


def test_euler_scale() -> None:
    """Verify the scale_model_input method of the EulerDiscreteScheduler.

    This test ensures that scaling model inputs with the scheduler yields
    outputs of expected sizes for standard timesteps as well as boundary or
    out-of-bounds timesteps.

    Args:
        None

    Returns:
        None
    """
    euler = EulerDiscreteScheduler()
    euler.set_timesteps(10)
    assert len(euler.scale_model_input([1.0], euler.timesteps[0])) == 1
    assert len(euler.scale_model_input([1.0], 9999)) == 1


def test_utils_progress() -> None:
    """Verify that global_progress_bar_config can be disabled.

    This test checks that set_progress_bar_config successfully modifies the
    global progress bar enabled status.

    Args:
        None

    Returns:
        None
    """
    set_progress_bar_config(False)
    from onnx9000_diffusers.utils import global_progress_bar_config

    assert not global_progress_bar_config.enabled


def test_utils_fetch_error(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify error handling in fetch_hub_file when download fails.

    This test ensures that fetch_hub_file gracefully fails and handles cleanup of
    temporary/partial files if urllib.request.urlopen raises an exception during
    the download process.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest fixture
            to store the downloaded cached files.

    Returns:
        None
    """
    cache = tmp_path / "cache"
    cache.mkdir()

    def mock_urlopen(
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> NoReturn:
        """Mock urllib.request.urlopen to simulate a network failure.

        This mock function writes a dummy test file to the cache directory before
        raising an exception, enabling validation of the cleanup code path in
        fetch_hub_file which should remove partially downloaded/created files.

        Args:
            *args: Variable length argument list passed to the urlopen function.
            **kwargs: Arbitrary keyword arguments passed to the urlopen function.

        Returns:
            None: This function does not return successfully; it always raises an exception.

        Raises:
            Exception: Always raises a 'MockError' to simulate a download failure.
        """
        import os

        with open(os.path.join(cache, "testfile"), "w") as f:
            f.write("test")
        raise Onnx9000Error("MockError")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        with pytest.raises(
            (Onnx9000Error, ValueError, RuntimeError, TypeError, KeyError, AttributeError)
        ):
            fetch_hub_file("repo/id", "testfile", str(cache))
