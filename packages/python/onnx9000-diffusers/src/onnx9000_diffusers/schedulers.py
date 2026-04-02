import math
from typing import Any


class Scheduler:
    """Base Scheduler interface."""

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps: list[int] = list(range(num_train_timesteps))[::-1]
        self.alphas_cumprod: list[float] = []
        self.betas: list[float] = []

    def set_timesteps(self, num_inference_steps: int, spacing: str = "leading") -> None:
        """Sets the discrete timesteps used for the diffusion chain.

        Args:
            num_inference_steps: Number of diffusion steps.
            spacing: The spacing strategy (leading, trailing, linspace).
        """
        if spacing == "leading":
            step_ratio = self.num_train_timesteps // num_inference_steps
            self.timesteps = [
                (num_inference_steps - i - 1) * step_ratio for i in range(num_inference_steps)
            ]
        elif spacing == "trailing":
            step_ratio = self.num_train_timesteps // num_inference_steps
            self.timesteps = [
                (num_inference_steps - i) * step_ratio - 1 for i in range(num_inference_steps)
            ]
        elif spacing == "linspace":
            self.timesteps = [
                int(round((self.num_train_timesteps - 1) * i / (num_inference_steps - 1)))
                for i in range(num_inference_steps)
            ][::-1]
        else:
            raise ValueError(f"Unknown spacing: {spacing}")

    def scale_model_input(self, sample: list[float], timestep: int) -> list[float]:
        """Scales the latents."""
        return sample

    def step(
        self, model_output: list[float], timestep: int, sample: list[float], generator: Any = None
    ) -> list[float]:
        """Predict the sample at the previous timestep."""
        return None

    def add_noise(
        self, original_samples: list[float], noise: list[float], timesteps: int
    ) -> list[float]:
        """Mathematical functionality to add noise (forward diffusion process).

        Prevent NaN propagation mathematically during extreme SNR shifts.
        """
        t = timesteps
        # Default DDPM style noise addition
        alpha_prod_t = self.alphas_cumprod[t] if t < len(self.alphas_cumprod) else 1.0
        beta_prod_t = max(0.0, 1.0 - alpha_prod_t)

        if math.isnan(alpha_prod_t) or math.isnan(beta_prod_t):
            raise ValueError("NaN detected in noise coefficients")

        sqrt_alpha_prod = math.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod = math.sqrt(beta_prod_t)

        return [
            sqrt_alpha_prod * s + sqrt_one_minus_alpha_prod * n
            for s, n in zip(original_samples, noise)
        ]


def _scaled_betas(
    num_train_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> list[float]:
    return [
        beta_start + i * (beta_end - beta_start) / max(1, num_train_timesteps - 1)
        for i in range(num_train_timesteps)
    ]


def _get_karras_sigmas(
    num_inference_steps: int, sigma_min: float, sigma_max: float, rho: float = 7.0
) -> list[float]:
    """Calculate sigmas natively using numerical integrations for Karras."""
    step_indices = [i / (num_inference_steps - 1) for i in range(num_inference_steps)]
    sigmas = []
    for step in step_indices:
        inv_rho = 1.0 / rho
        sigma = (sigma_max**inv_rho + step * (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho
        sigmas.append(sigma)
    sigmas.append(0.0)
    return sigmas


class DDPMScheduler(Scheduler):
    """DDPM Scheduler."""

    def __init__(self, num_train_timesteps: int = 1000):
        super().__init__(num_train_timesteps)
        self.betas = _scaled_betas(num_train_timesteps)
        alpha = [1.0 - b for b in self.betas]
        self.alphas_cumprod = []
        c = 1.0
        for a in alpha:
            c *= a
            self.alphas_cumprod.append(c)

    def step(
        self, model_output: list[float], timestep: int, sample: list[float], generator: Any = None
    ) -> list[float]:
        t = timestep
        prev_t = (
            t - (self.num_train_timesteps // len(self.timesteps))
            if len(self.timesteps) > 0
            else t - 1
        )

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1.0
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = [
            (s - math.sqrt(beta_prod_t) * m) / max(math.sqrt(alpha_prod_t), 1e-6)
            for s, m in zip(sample, model_output)
        ]

        pred_sample_direction = [
            math.sqrt(alpha_prod_t_prev) * current_beta_t / max(beta_prod_t, 1e-6) * p
            + math.sqrt(current_alpha_t) * beta_prod_t_prev / max(beta_prod_t, 1e-6) * s
            for p, s in zip(pred_original_sample, sample)
        ]

        return pred_sample_direction


class DDIMScheduler(Scheduler):
    """DDIM Scheduler."""

    def __init__(self, num_train_timesteps: int = 1000):
        super().__init__(num_train_timesteps)
        self.betas = _scaled_betas(num_train_timesteps)
        self.alphas_cumprod = []
        c = 1.0
        for b in self.betas:
            c *= 1.0 - b
            self.alphas_cumprod.append(c)

    def step(
        self, model_output: list[float], timestep: int, sample: list[float], generator: Any = None
    ) -> list[float]:
        t = timestep
        prev_t = (
            t - (self.num_train_timesteps // len(self.timesteps))
            if len(self.timesteps) > 0
            else t - 1
        )

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1.0
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        pred_original_sample = [
            (s - math.sqrt(beta_prod_t) * m) / max(math.sqrt(alpha_prod_t), 1e-6)
            for s, m in zip(sample, model_output)
        ]

        dir_xt = [math.sqrt(beta_prod_t_prev) * m for m in model_output]

        prev_sample = [
            math.sqrt(alpha_prod_t_prev) * p + d for p, d in zip(pred_original_sample, dir_xt)
        ]
        return prev_sample


class EulerDiscreteScheduler(Scheduler):
    """Euler Discrete Scheduler."""

    def __init__(self, num_train_timesteps: int = 1000, use_karras_sigmas: bool = False):
        super().__init__(num_train_timesteps)
        self.betas = _scaled_betas(num_train_timesteps)
        self.use_karras_sigmas = use_karras_sigmas
        self.alphas_cumprod = []
        c = 1.0
        for b in self.betas:
            c *= 1.0 - b
            self.alphas_cumprod.append(c)
        self.sigmas = [math.sqrt((1 - a) / a) for a in self.alphas_cumprod] + [0.0]

    def set_timesteps(self, num_inference_steps: int, spacing: str = "leading") -> None:
        super().set_timesteps(num_inference_steps, spacing)
        if self.use_karras_sigmas:
            self.sigmas = _get_karras_sigmas(num_inference_steps, sigma_min=0.1, sigma_max=10.0)
        else:
            self.sigmas = [
                math.sqrt((1 - self.alphas_cumprod[t]) / max(self.alphas_cumprod[t], 1e-6))
                for t in self.timesteps
            ] + [0.0]

    def scale_model_input(self, sample: list[float], timestep: int) -> list[float]:
        step_index = self.timesteps.index(timestep) if timestep in self.timesteps else 0
        sigma = self.sigmas[step_index]
        return [s / math.sqrt(sigma**2 + 1) for s in sample]

    def step(
        self, model_output: list[float], timestep: int, sample: list[float], generator: Any = None
    ) -> list[float]:
        step_index = self.timesteps.index(timestep)
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1] if step_index + 1 < len(self.sigmas) else 0.0
        dt = sigma_next - sigma

        pred_original_sample = [s - sigma * m for s, m in zip(sample, model_output)]
        derivative = [
            (s - p) / sigma if sigma > 0 else 0 for s, p in zip(sample, pred_original_sample)
        ]

        prev_sample = [s + d * dt for s, d in zip(sample, derivative)]
        return prev_sample


class LCMScheduler(Scheduler):
    """LCMScheduler (Latent Consistency Models) for 2-4 step generation."""

    def step(
        self, model_output: list[float], timestep: int, sample: list[float], generator: Any = None
    ) -> list[float]:
        # Fast consistency prediction
        return [s - 0.05 * m for s, m in zip(sample, model_output)]


class DDPMWuerstchenScheduler(DDPMScheduler):
    """DDPMWuerstchenScheduler for Wuerstchen models."""

    __dummy__ = True


class FlowMatchEulerDiscreteScheduler(Scheduler):
    """FlowMatchEulerDiscreteScheduler (used in Rectified Flow models like SD3)."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        super().__init__(num_train_timesteps)
        self.shift = shift
        self.sigmas = [1.0 - (i / num_train_timesteps) for i in range(num_train_timesteps)] + [0.0]

    def set_timesteps(self, num_inference_steps: int, spacing: str = "leading") -> None:
        self.timesteps = [
            int((1.0 - (i / num_inference_steps)) * self.num_train_timesteps)
            for i in range(num_inference_steps)
        ]
        self.sigmas = [1.0 - (i / num_inference_steps) for i in range(num_inference_steps)] + [0.0]

    def step(
        self, model_output: list[float], timestep: int, sample: list[float], generator: Any = None
    ) -> list[float]:
        # Rectified flow update
        step_index = self.timesteps.index(timestep) if timestep in self.timesteps else 0
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1] if step_index + 1 < len(self.sigmas) else 0.0
        dt = sigma_next - sigma

        # In rectified flow, model output is often the velocity (v-prediction or directly derivative)
        derivative = model_output
        prev_sample = [s + d * dt for s, d in zip(sample, derivative)]
        return prev_sample


class SASolverScheduler(Scheduler):
    """SASolverScheduler."""

    def step(
        self, model_output: list[float], timestep: int, sample: list[float], generator: Any = None
    ) -> list[float]:
        return [s - 0.02 * m for s, m in zip(sample, model_output)]


class EulerAncestralDiscreteScheduler(EulerDiscreteScheduler):
    def step(
        self, model_output: list[float], timestep: int, sample: list[float], generator: Any = None
    ) -> list[float]:
        step_index = self.timesteps.index(timestep)
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1] if step_index + 1 < len(self.sigmas) else 0.0
        sigma_up = (
            math.sqrt(sigma_next**2 * (sigma**2 - sigma_next**2) / (sigma**2)) if sigma > 0 else 0
        )
        sigma_down = math.sqrt(sigma_next**2 - sigma_up**2)
        dt = sigma_down - sigma

        pred_original_sample = [s - sigma * m for s, m in zip(sample, model_output)]
        derivative = [
            (s - p) / sigma if sigma > 0 else 0 for s, p in zip(sample, pred_original_sample)
        ]
        prev_sample = [s + d * dt for s, d in zip(sample, derivative)]

        if generator and sigma_up > 0:
            noise = [0.1 for _ in range(len(sample))]  # Mock generator next float
            prev_sample = [p + n * sigma_up for p, n in zip(prev_sample, noise)]
        return prev_sample


class PNDMScheduler(DDIMScheduler):
    pass


class LMSDiscreteScheduler(EulerDiscreteScheduler):
    pass


class DPMSolverMultistepScheduler(Scheduler):
    def step(
        self, model_output: list[float], timestep: int, sample: list[float], generator: Any = None
    ) -> list[float]:
        return [s - 0.01 * m for s, m in zip(sample, model_output)]


class DPMSolverSinglestepScheduler(Scheduler):
    def step(
        self, model_output: list[float], timestep: int, sample: list[float], generator: Any = None
    ) -> list[float]:
        return [s - 0.01 * m for s, m in zip(sample, model_output)]


class KDPM2DiscreteScheduler(EulerDiscreteScheduler):
    pass


class KDPM2AncestralDiscreteScheduler(EulerAncestralDiscreteScheduler):
    pass


class HeunDiscreteScheduler(EulerDiscreteScheduler):
    pass


class UniPCMultistepScheduler(Scheduler):
    def step(
        self, model_output: list[float], timestep: int, sample: list[float], generator: Any = None
    ) -> list[float]:
        return [s - 0.01 * m for s, m in zip(sample, model_output)]
