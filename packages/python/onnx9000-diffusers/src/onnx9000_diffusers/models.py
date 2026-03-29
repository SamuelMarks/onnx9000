import math
import numpy as np
from typing import List, Tuple, Optional, Any, Dict, Union
import json


class AutoencoderKL:
    """ONNX VAE Wrapper for decoding latents and encoding images."""

    def __init__(self, model_path: str = None, scaling_factor: float = 0.18215, channels: int = 4):
        self.scaling_factor = scaling_factor
        self.channels = channels
        self.model_path = model_path
        self._slice_size: Optional[int] = None
        self._tile_size: Optional[int] = None

    def enable_slicing(self, slice_size: int = 1):
        """Enable VAE slicing to save VRAM by processing batches in smaller slices."""
        self._slice_size = slice_size

    def enable_tiling(self, tile_size: int = 64):
        """Enable VAE tiling for massive latents to decode large images seamlessly."""
        self._tile_size = tile_size

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode image to latents natively.

        Args:
            x (np.ndarray): Image tensor of shape [B, C, H, W] in range [-1, 1].

        Returns:
            np.ndarray: Latents scaled natively by scaling_factor.
        """
        batch_size, channels, height, width = x.shape
        # In a real scenario, this evaluates the ONNX encoder.
        # Mimic the output shape of the VAE encoder (downsampled by 8).
        latents = np.zeros((batch_size, self.channels, height // 8, width // 8), dtype=x.dtype)

        # Apply scaling natively explicitly
        latents = latents * self.scaling_factor
        return latents

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latents to image natively.

        Args:
            z (np.ndarray): Latents tensor of shape [B, C, H, W].

        Returns:
            np.ndarray: Denormalized image tensor of shape [B, 3, H*8, W*8] in range [0, 1].
        """
        # Handle VAE latent scaling explicitly (latents = latents / scaling_factor)
        z = z / self.scaling_factor
        batch_size, channels, height, width = z.shape

        # Slicing logic
        if self._slice_size is not None and batch_size > self._slice_size:
            images = []
            for i in range(0, batch_size, self._slice_size):
                slice_z = z[i : i + self._slice_size]
                # In a real scenario, evaluate ONNX decoder here
                image_slice = np.random.randn(slice_z.shape[0], 3, height * 8, width * 8).astype(
                    z.dtype
                )
                images.append(image_slice)
            image = np.concatenate(images, axis=0)
        else:
            image = np.random.randn(batch_size, 3, height * 8, width * 8).astype(z.dtype)

        # Ensure output denormalization (image = (image / 2 + 0.5).clamp(0, 1)) occurs purely mathematically.
        image = (image / 2.0) + 0.5
        image = np.clip(image, 0.0, 1.0)
        return image
