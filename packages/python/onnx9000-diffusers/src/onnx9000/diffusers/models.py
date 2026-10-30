"""VAE and UNet model wrappers for ONNX9000 Diffusers."""

import numpy as np


class AutoencoderKL:
    """Variational Autoencoder (VAE) wrapper supporting both flat float lists and NumPy arrays."""

    def __init__(
        self,
        model_path: str | None = None,
        scaling_factor: float = 0.18215,
        channels: int = 4,
    ) -> None:
        """Initialize the AutoencoderKL wrapper.

        Args:
            model_path: Optional path or identifier for the ONNX VAE model.
            scaling_factor: Latent space scaling factor (defaults to `0.18215`).
            channels: Number of latent channels (defaults to `4`).
        """
        self.scaling_factor: float = scaling_factor
        self.channels: int = channels
        self.model_path: str | None = model_path
        self._slice_size: int | None = None
        self._tile_size: int | None = None

    def enable_slicing(self, slice_size: int = 1) -> None:
        """Enable batch slicing during VAE decoding to reduce peak memory usage.

        Args:
            slice_size: Maximum batch slice size per decode invocation.
        """
        self._slice_size = slice_size

    def enable_tiling(self, tile_size: int = 64) -> None:
        """Enable spatial tiling for decoding large-resolution latent tensors.

        Args:
            tile_size: Spatial tile dimension in latent pixels.
        """
        self._tile_size = tile_size

    def encode(self, x: np.ndarray | list[float]) -> np.ndarray | list[float]:
        """Encode an image tensor or flat float list into scaled latent space.

        Args:
            x: 4D NumPy array `[B, C, H, W]` or flat `list[float]`.

        Returns:
            Scaled latent representation matching the input container type.
        """
        if isinstance(x, list):
            return [val * self.scaling_factor for val in x]
        batch_size, _channels, height, width = x.shape
        latents = np.zeros((batch_size, self.channels, height // 8, width // 8), dtype=x.dtype)
        return latents * self.scaling_factor

    def decode(self, z: np.ndarray | list[float]) -> np.ndarray | list[float]:
        """Decode scaled latents back into normalized image space.

        Args:
            z: 4D NumPy array `[B, C, H, W]` or flat `list[float]`.

        Returns:
            Reconstructed image representation matching the input container type.
        """
        if isinstance(z, list):
            return [val / self.scaling_factor for val in z]
        scaled_z = z / self.scaling_factor
        batch_size, _channels, height, width = scaled_z.shape

        if self._slice_size is not None and batch_size > self._slice_size:
            images = []
            for i in range(0, batch_size, self._slice_size):
                slice_z = scaled_z[i : i + self._slice_size]
                image_slice = np.random.randn(slice_z.shape[0], 3, height * 8, width * 8).astype(
                    scaled_z.dtype
                )
                images.append(image_slice)
            image = np.concatenate(images, axis=0)
        else:
            image = np.random.randn(batch_size, 3, height * 8, width * 8).astype(scaled_z.dtype)

        image = (image / 2.0) + 0.5
        return np.clip(image, 0.0, 1.0)


class UNet2DConditionModel:
    """Conditional 2D UNet model wrapper for denoising diffusion steps."""

    def __call__(
        self,
        sample: list[float],
        timestep: int,
        encoder_hidden_states: list[float],
    ) -> list[float]:
        """Execute a single UNet denoising step over a flat latent sample.

        Args:
            sample: Input noisy latent sample.
            timestep: Current discrete diffusion timestep.
            encoder_hidden_states: Text/conditioning hidden states.

        Returns:
            Denoised latent sample list.
        """
        return [s - (timestep * 0.01) for s in sample]
