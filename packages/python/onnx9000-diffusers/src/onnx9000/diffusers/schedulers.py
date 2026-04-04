"""Schedulers for diffusion-based denoising processes."""

import math


class Scheduler:
    """Base class for all diffusion schedulers."""

    def __init__(self, num_train_timesteps: int = 1000):
        """Initialize the scheduler.

        Args:
            num_train_timesteps: Total number of training timesteps.

        """
        self.num_train_timesteps: int = num_train_timesteps
        self.timesteps: list[int] = list(range(num_train_timesteps - 1, -1, -1))

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set the timesteps for the denoising process.

        Args:
            num_inference_steps: Number of inference steps to take.

        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = [i * step_ratio for i in range(num_inference_steps)][::-1]


class DDIMScheduler(Scheduler):
    """Denoising Diffusion Implicit Models (DDIM) scheduler."""

    def step(self, model_output: list[float], timestep: int, sample: list[float]) -> list[float]:
        """Compute the previous sample using the DDIM step.

        Args:
            model_output: Predicted noise or model output.
            timestep: Current timestep.
            sample: Current sample.

        Returns:
            The denoised sample at the previous timestep.

        """
        alpha_prod_t = 1.0 - (timestep / self.num_train_timesteps)
        beta_prod_t = 1 - alpha_prod_t
        return [
            (s - math.sqrt(beta_prod_t) * o) / math.sqrt(alpha_prod_t)
            for s, o in zip(sample, model_output)
        ]


class DDPMScheduler(Scheduler):
    """Denoising Diffusion Probabilistic Models (DDPM) scheduler."""

    def step(self, model_output: list[float], timestep: int, sample: list[float]) -> list[float]:
        """Compute the previous sample using the DDPM step.

        Args:
            model_output: Predicted noise.
            timestep: Current timestep.
            sample: Current sample.

        Returns:
            The denoised sample.

        """
        alpha_t = 1.0 - (timestep / self.num_train_timesteps)
        return [(s - (1 - alpha_t) * o) / math.sqrt(alpha_t) for s, o in zip(sample, model_output)]


class EulerDiscreteScheduler(Scheduler):
    """Euler Discrete Scheduler implementation."""

    def step(self, model_output: list[float], timestep: int, sample: list[float]) -> list[float]:
        """Compute the previous sample using a single Euler step.

        Args:
            model_output: Predicted noise.
            timestep: Current timestep.
            sample: Current sample.

        Returns:
            The updated sample.

        """
        sigma = timestep / self.num_train_timesteps
        return [s + o * sigma for s, o in zip(sample, model_output)]


class LCMScheduler(Scheduler):
    """Latent Consistency Models (LCM) scheduler for fast inference."""

    def step(self, model_output: list[float], timestep: int, sample: list[float]) -> list[float]:
        """Compute the previous sample for LCM.

        Args:
            model_output: Predicted latent.
            timestep: Current timestep.
            sample: Current sample.

        Returns:
            The consistent latent sample.

        """
        return [s - o for s, o in zip(sample, model_output)]
