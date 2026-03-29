import numpy as np
import pytest
from onnx9000_diffusers.models import AutoencoderKL


def test_autoencoderkl_wrapper():
    vae = AutoencoderKL(scaling_factor=0.18215, channels=4)
    # Test Encoding
    img = np.random.randn(2, 3, 512, 512).astype(np.float32)
    latents = vae.encode(img)
    assert latents.shape == (2, 4, 64, 64)

    # Test Decoding
    latents = np.random.randn(2, 4, 64, 64).astype(np.float32)
    decoded = vae.decode(latents)
    assert decoded.shape == (2, 3, 512, 512)
    assert np.all(decoded >= 0.0) and np.all(decoded <= 1.0)


def test_autoencoderkl_slicing():
    vae = AutoencoderKL(scaling_factor=0.18215, channels=4)
    vae.enable_slicing(slice_size=1)
    latents = np.random.randn(3, 4, 64, 64).astype(np.float32)
    decoded = vae.decode(latents)
    assert decoded.shape == (3, 3, 512, 512)
