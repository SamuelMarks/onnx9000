"""Models for diffusion-based generative tasks."""


class AutoencoderKL:
    """Variational Autoencoder (VAE) for mapping images to latents and vice versa."""

    def encode(self, x: list[float]) -> list[float]:
        """Map VAE encode (Image -> Latent) operations natively.

        Args:
            x: Input image data as a flat list of floats.

        Returns:
            Latent representation.
        """
        return [val * 0.18215 for val in x]

    def decode(self, x: list[float]) -> list[float]:
        """Map VAE decode (Latent -> Image) operations natively.

        Args:
            x: Latent representation as a flat list of floats.

        Returns:
            Reconstructed image data.
        """
        return [(val / 0.18215) for val in x]


class UNet2DConditionModel:
    """Conditional UNet model for denoising diffusion steps."""

    def __call__(
        self, sample: list[float], timestep: int, encoder_hidden_states: list[float]
    ) -> list[float]:
        """Simulate UNet denoising step natively.

        Args:
            sample: Input latent sample.
            timestep: Current denoising timestep.
            encoder_hidden_states: Conditioning states from text encoder.

        Returns:
            Denoised latent sample.
        """
        return [s - (timestep * 0.01) for s in sample]
