from typing import List


class AutoencoderKL:
    def encode(self, x: List[float]) -> List[float]:
        """Map VAE encode (Image -> Latent) operations natively."""
        return [val * 0.18215 for val in x]

    def decode(self, x: List[float]) -> List[float]:
        """Map VAE decode (Latent -> Image) operations natively."""
        return [(val / 0.18215) for val in x]


class UNet2DConditionModel:
    def __call__(
        self, sample: List[float], timestep: int, encoder_hidden_states: List[float]
    ) -> List[float]:
        """Simulate UNet denoising step natively."""
        return [s - (timestep * 0.01) for s in sample]
