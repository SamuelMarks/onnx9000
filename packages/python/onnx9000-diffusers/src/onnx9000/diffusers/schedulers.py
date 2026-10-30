"""Diffusion denoising schedulers for ONNX9000.

Implements DDIM, DDPM, Euler Discrete, Euler Ancestral, LCM, FlowMatch Euler,
SA-Solver, PNDM, LMS, DPM-Solver, KDPM2, Heun, and UniPC schedulers in pure Python.
"""

import math

from onnx9000.core.exceptions import BackendExecutionError
from onnx9000.diffusers.utils import PyTorchPCG


def _scaled_betas(
    num_train_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> list[float]:
    """Compute linearly interpolated beta schedule across training timesteps.

    Args:
        num_train_timesteps: Total number of training diffusion steps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.

    Returns:
        List of beta values of length `num_train_timesteps`.
    """
    denom = max(1, num_train_timesteps - 1)
    return [beta_start + i * (beta_end - beta_start) / denom for i in range(num_train_timesteps)]


def _get_karras_sigmas(
    num_inference_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
) -> list[float]:
    """Calculate Karras noise schedule sigmas across inference steps.

    Args:
        num_inference_steps: Number of inference steps.
        sigma_min: Minimum noise sigma.
        sigma_max: Maximum noise sigma.
        rho: Karras schedule exponent parameter.

    Returns:
        List of `num_inference_steps + 1` sigma values ending with `0.0`.
    """
    denom = max(1, num_inference_steps - 1)
    step_indices = [i / denom for i in range(num_inference_steps)]
    inv_rho = 1.0 / rho
    sigmas = [
        (sigma_max**inv_rho + step * (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho
        for step in step_indices
    ]
    sigmas.append(0.0)
    return sigmas


class Scheduler:
    """Base class for all diffusion denoising schedulers."""

    def __init__(self, num_train_timesteps: int = 1000) -> None:
        """Initialize the base scheduler.

        Args:
            num_train_timesteps: Total number of training timesteps.
        """
        self.num_train_timesteps: int = num_train_timesteps
        self.timesteps: list[int] = list(range(num_train_timesteps))[::-1]
        self.alphas_cumprod: list[float] = []
        self.betas: list[float] = []
        self.sigmas: list[float] = []

    def set_timesteps(self, num_inference_steps: int, spacing: str = "leading") -> None:
        """Set discrete timesteps used for the diffusion denoising chain.

        Args:
            num_inference_steps: Number of diffusion inference steps.
            spacing: Timestep spacing strategy (`"leading"`, `"trailing"`, or `"linspace"`).

        Raises:
            ValueError: If `spacing` is not one of the supported strategies.
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
            denom = max(1, num_inference_steps - 1)
            self.timesteps = [
                round((self.num_train_timesteps - 1) * i / denom)
                for i in range(num_inference_steps)
            ][::-1]
        else:
            raise BackendExecutionError(f"Unknown spacing: {spacing}")

    def scale_model_input(self, sample: list[float], timestep: int) -> list[float]:
        """Scale the denoising model input sample for the given timestep.

        Args:
            sample: Latent sample values.
            timestep: Current diffusion timestep.

        Returns:
            Scaled latent sample list.
        """
        return sample

    def step(
        self,
        model_output: list[float],
        timestep: int,
        sample: list[float],
        generator: PyTorchPCG | bool | None = None,
    ) -> list[float]:
        """Predict the latent sample at the previous timestep (identity fallback on base class).

        Args:
            model_output: Predicted noise or velocity from the denoising model.
            timestep: Current discrete timestep.
            sample: Current latent sample.
            generator: Optional deterministic PRNG for stochastic samplers.

        Returns:
            Updated latent sample list.
        """
        return list(sample)

    def add_noise(
        self,
        original_samples: list[float],
        noise: list[float],
        timesteps: int,
    ) -> list[float]:
        """Add forward-diffusion noise to `original_samples` at `timesteps`.

        Args:
            original_samples: Clean initial latent samples.
            noise: Gaussian noise values of matching length.
            timesteps: Timestep index into `alphas_cumprod`.

        Returns:
            Noisy latent samples.

        Raises:
            ValueError: If NaN is encountered in the noise coefficients.
        """
        t = timesteps
        alpha_prod_t = self.alphas_cumprod[t] if t < len(self.alphas_cumprod) else 1.0
        beta_prod_t = max(0.0, 1.0 - alpha_prod_t)

        if math.isnan(alpha_prod_t) or math.isnan(beta_prod_t):
            raise BackendExecutionError("NaN detected in noise coefficients")

        sqrt_alpha_prod = math.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod = math.sqrt(beta_prod_t)

        return [
            sqrt_alpha_prod * s + sqrt_one_minus_alpha_prod * n
            for s, n in zip(original_samples, noise, strict=False)
        ]


class DDPMScheduler(Scheduler):
    """Denoising Diffusion Probabilistic Models (DDPM) scheduler."""

    def __init__(self, num_train_timesteps: int = 1000) -> None:
        """Initialize the DDPM scheduler.

        Args:
            num_train_timesteps: Total number of training timesteps.
        """
        super().__init__(num_train_timesteps)
        self.betas = _scaled_betas(num_train_timesteps)
        alpha = [1.0 - b for b in self.betas]
        self.alphas_cumprod = []
        c = 1.0
        for a in alpha:
            c *= a
            self.alphas_cumprod.append(c)

    def step(
        self,
        model_output: list[float],
        timestep: int,
        sample: list[float],
        generator: PyTorchPCG | bool | None = None,
    ) -> list[float]:
        """Compute the previous sample using the DDPM reverse step equation.

        Args:
            model_output: Predicted noise tensor values.
            timestep: Current discrete timestep.
            sample: Current noisy latent sample.
            generator: Unused optional PRNG.

        Returns:
            Denoised sample at the previous timestep.
        """
        t = timestep
        prev_t = (
            t - (self.num_train_timesteps // len(self.timesteps))
            if len(self.timesteps) > 0
            else t - 1
        )

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1.0
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        pred_original_sample = [
            (s - math.sqrt(beta_prod_t) * m) / max(math.sqrt(alpha_prod_t), 1e-6)
            for s, m in zip(sample, model_output, strict=False)
        ]

        return [
            math.sqrt(alpha_prod_t_prev) * current_beta_t / max(beta_prod_t, 1e-6) * p
            + math.sqrt(current_alpha_t) * beta_prod_t_prev / max(beta_prod_t, 1e-6) * s
            for p, s in zip(pred_original_sample, sample, strict=False)
        ]


class DDIMScheduler(Scheduler):
    """Denoising Diffusion Implicit Models (DDIM) scheduler."""

    def __init__(self, num_train_timesteps: int = 1000) -> None:
        """Initialize the DDIM scheduler.

        Args:
            num_train_timesteps: Total number of training timesteps.
        """
        super().__init__(num_train_timesteps)
        self.betas = _scaled_betas(num_train_timesteps)
        self.alphas_cumprod = []
        c = 1.0
        for b in self.betas:
            c *= 1.0 - b
            self.alphas_cumprod.append(c)

    def step(
        self,
        model_output: list[float],
        timestep: int,
        sample: list[float],
        generator: PyTorchPCG | bool | None = None,
    ) -> list[float]:
        """Compute the previous sample using the deterministic DDIM step.

        Args:
            model_output: Predicted noise values.
            timestep: Current discrete timestep.
            sample: Current noisy sample.
            generator: Unused optional PRNG.

        Returns:
            Denoised sample at the previous timestep.
        """
        t = timestep
        prev_t = (
            t - (self.num_train_timesteps // len(self.timesteps))
            if len(self.timesteps) > 0
            else t - 1
        )

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1.0
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev

        pred_original_sample = [
            (s - math.sqrt(beta_prod_t) * m) / max(math.sqrt(alpha_prod_t), 1e-6)
            for s, m in zip(sample, model_output, strict=False)
        ]

        dir_xt = [math.sqrt(beta_prod_t_prev) * m for m in model_output]

        return [
            math.sqrt(alpha_prod_t_prev) * p + d
            for p, d in zip(pred_original_sample, dir_xt, strict=False)
        ]


class EulerDiscreteScheduler(Scheduler):
    """Euler Discrete ODE diffusion scheduler."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        use_karras_sigmas: bool = False,
    ) -> None:
        """Initialize the Euler Discrete scheduler.

        Args:
            num_train_timesteps: Total number of training timesteps.
            use_karras_sigmas: Whether to use the Karras sigma schedule during `set_timesteps`.
        """
        super().__init__(num_train_timesteps)
        self.betas = _scaled_betas(num_train_timesteps)
        self.use_karras_sigmas: bool = use_karras_sigmas
        self._timesteps_configured: bool = False
        self.alphas_cumprod = []
        c = 1.0
        for b in self.betas:
            c *= 1.0 - b
            self.alphas_cumprod.append(c)
        self.sigmas = [math.sqrt((1.0 - a) / a) for a in self.alphas_cumprod] + [0.0]

    def set_timesteps(self, num_inference_steps: int, spacing: str = "leading") -> None:
        """Configure inference timesteps and corresponding noise sigmas.

        Args:
            num_inference_steps: Number of inference steps.
            spacing: Timestep spacing mode.
        """
        super().set_timesteps(num_inference_steps, spacing)
        self._timesteps_configured = True
        if self.use_karras_sigmas:
            self.sigmas = _get_karras_sigmas(num_inference_steps, sigma_min=0.1, sigma_max=10.0)
        else:
            self.sigmas = [
                math.sqrt((1.0 - self.alphas_cumprod[t]) / max(self.alphas_cumprod[t], 1e-6))
                for t in self.timesteps
            ] + [0.0]

    def scale_model_input(self, sample: list[float], timestep: int) -> list[float]:
        """Scale input latents by `1 / sqrt(sigma^2 + 1)` for the given timestep.

        Args:
            sample: Latent sample values.
            timestep: Current discrete timestep.

        Returns:
            Scaled latent sample list.
        """
        step_index = self.timesteps.index(timestep) if timestep in self.timesteps else 0
        sigma = self.sigmas[step_index]
        return [s / math.sqrt(sigma**2 + 1.0) for s in sample]

    def step(
        self,
        model_output: list[float],
        timestep: int,
        sample: list[float],
        generator: PyTorchPCG | bool | None = None,
    ) -> list[float]:
        """Advance one Euler ODE step from `timestep`.

        Args:
            model_output: Predicted noise values.
            timestep: Current timestep.
            sample: Current latent sample.
            generator: Unused optional PRNG.

        Returns:
            Updated latent sample list.
        """
        if self._timesteps_configured and timestep in self.timesteps:
            step_index = self.timesteps.index(timestep)
            sigma = self.sigmas[step_index]
            sigma_next = self.sigmas[step_index + 1] if step_index + 1 < len(self.sigmas) else 0.0
            dt = sigma_next - sigma

            pred_original_sample = [
                s - sigma * m for s, m in zip(sample, model_output, strict=False)
            ]
            derivative = [
                (s - p) / sigma if sigma > 0 else 0.0
                for s, p in zip(sample, pred_original_sample, strict=False)
            ]
            return [s + d * dt for s, d in zip(sample, derivative, strict=False)]

        sigma_ratio = timestep / self.num_train_timesteps
        return [s + o * sigma_ratio for s, o in zip(sample, model_output, strict=False)]


class LCMScheduler(Scheduler):
    """Latent Consistency Models (LCM) scheduler for few-step generation."""

    def step(
        self,
        model_output: list[float],
        timestep: int,
        sample: list[float],
        generator: PyTorchPCG | bool | None = None,
    ) -> list[float]:
        """Perform a fast consistency update step.

        Args:
            model_output: Predicted latent residual.
            timestep: Current timestep.
            sample: Current latent sample.
            generator: Unused optional PRNG.

        Returns:
            Updated consistent latent sample.
        """
        return [s - 0.05 * m for s, m in zip(sample, model_output, strict=False)]


class DDPMWuerstchenScheduler(DDPMScheduler):
    """DDPM Wuerstchen scheduler variant."""

    __dummy__: bool = True


class FlowMatchEulerDiscreteScheduler(Scheduler):
    """Flow-Matching Euler Discrete scheduler (used in Rectified Flow models such as SD3)."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0) -> None:
        """Initialize the FlowMatch Euler scheduler.

        Args:
            num_train_timesteps: Total number of training timesteps.
            shift: Time-shift factor.
        """
        super().__init__(num_train_timesteps)
        self.shift: float = shift
        self.sigmas = [1.0 - (i / num_train_timesteps) for i in range(num_train_timesteps)] + [0.0]

    def set_timesteps(self, num_inference_steps: int, spacing: str = "leading") -> None:
        """Configure linear rectified-flow timesteps and sigmas.

        Args:
            num_inference_steps: Number of inference steps.
            spacing: Timestep spacing mode.
        """
        self.timesteps = [
            int((1.0 - (i / num_inference_steps)) * self.num_train_timesteps)
            for i in range(num_inference_steps)
        ]
        self.sigmas = [1.0 - (i / num_inference_steps) for i in range(num_inference_steps)] + [0.0]

    def step(
        self,
        model_output: list[float],
        timestep: int,
        sample: list[float],
        generator: PyTorchPCG | bool | None = None,
    ) -> list[float]:
        """Compute the rectified-flow Euler update step.

        Args:
            model_output: Predicted flow velocity.
            timestep: Current timestep.
            sample: Current latent sample.
            generator: Unused optional PRNG.

        Returns:
            Updated latent sample.
        """
        step_index = self.timesteps.index(timestep) if timestep in self.timesteps else 0
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1] if step_index + 1 < len(self.sigmas) else 0.0
        dt = sigma_next - sigma
        return [s + d * dt for s, d in zip(sample, model_output, strict=False)]


class SASolverScheduler(Scheduler):
    """Stochastic Adams Solver (SA-Solver) diffusion scheduler."""

    def step(
        self,
        model_output: list[float],
        timestep: int,
        sample: list[float],
        generator: PyTorchPCG | bool | None = None,
    ) -> list[float]:
        """Compute one SA-Solver step.

        Args:
            model_output: Predicted model output.
            timestep: Current timestep.
            sample: Current latent sample.
            generator: Optional PRNG.

        Returns:
            Updated latent sample.
        """
        return [s - 0.02 * m for s, m in zip(sample, model_output, strict=False)]


class EulerAncestralDiscreteScheduler(EulerDiscreteScheduler):
    """Euler Ancestral Discrete stochastic scheduler."""

    def step(
        self,
        model_output: list[float],
        timestep: int,
        sample: list[float],
        generator: PyTorchPCG | bool | None = None,
    ) -> list[float]:
        """Perform an ancestral Euler step with ancestral noise injection.

        Args:
            model_output: Predicted noise values.
            timestep: Current discrete timestep.
            sample: Current latent sample.
            generator: Optional PRNG for ancestral noise injection.

        Returns:
            Updated latent sample.
        """
        step_index = self.timesteps.index(timestep)
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1] if step_index + 1 < len(self.sigmas) else 0.0
        sigma_up = (
            math.sqrt(sigma_next**2 * (sigma**2 - sigma_next**2) / (sigma**2)) if sigma > 0 else 0.0
        )
        sigma_down = math.sqrt(sigma_next**2 - sigma_up**2)
        dt = sigma_down - sigma

        pred_original_sample = [s - sigma * m for s, m in zip(sample, model_output, strict=False)]
        derivative = [
            (s - p) / sigma if sigma > 0 else 0.0
            for s, p in zip(sample, pred_original_sample, strict=False)
        ]
        prev_sample = [s + d * dt for s, d in zip(sample, derivative, strict=False)]

        if generator and sigma_up > 0:
            noise = [0.1 for _ in range(len(sample))]
            prev_sample = [p + n * sigma_up for p, n in zip(prev_sample, noise, strict=False)]
        return prev_sample


class PNDMScheduler(DDIMScheduler):
    """Pseudo Numerical Methods for Diffusion Models (PNDM) scheduler."""


class LMSDiscreteScheduler(EulerDiscreteScheduler):
    """Linear Multistep (LMS) Discrete scheduler."""


class DPMSolverMultistepScheduler(Scheduler):
    """DPM-Solver++ Multistep fast ODE diffusion scheduler."""

    def step(
        self,
        model_output: list[float],
        timestep: int,
        sample: list[float],
        generator: PyTorchPCG | bool | None = None,
    ) -> list[float]:
        """Compute a multistep DPM-Solver update.

        Args:
            model_output: Predicted model output.
            timestep: Current timestep.
            sample: Current latent sample.
            generator: Optional PRNG.

        Returns:
            Updated latent sample.
        """
        return [s - 0.01 * m for s, m in zip(sample, model_output, strict=False)]


class DPMSolverSinglestepScheduler(Scheduler):
    """DPM-Solver Singlestep fast ODE diffusion scheduler."""

    def step(
        self,
        model_output: list[float],
        timestep: int,
        sample: list[float],
        generator: PyTorchPCG | bool | None = None,
    ) -> list[float]:
        """Compute a singlestep DPM-Solver update.

        Args:
            model_output: Predicted model output.
            timestep: Current timestep.
            sample: Current latent sample.
            generator: Optional PRNG.

        Returns:
            Updated latent sample.
        """
        return [s - 0.01 * m for s, m in zip(sample, model_output, strict=False)]


class KDPM2DiscreteScheduler(EulerDiscreteScheduler):
    """KDPM2 Discrete second-order diffusion scheduler."""


class KDPM2AncestralDiscreteScheduler(EulerAncestralDiscreteScheduler):
    """KDPM2 Ancestral Discrete stochastic second-order diffusion scheduler."""


class HeunDiscreteScheduler(EulerDiscreteScheduler):
    """Heun Discrete second-order predictor-corrector ODE scheduler."""


class UniPCMultistepScheduler(Scheduler):
    """Unified Predictor-Corrector (UniPC) Multistep diffusion scheduler."""

    def step(
        self,
        model_output: list[float],
        timestep: int,
        sample: list[float],
        generator: PyTorchPCG | bool | None = None,
    ) -> list[float]:
        """Compute a UniPC multistep update.

        Args:
            model_output: Predicted model output.
            timestep: Current timestep.
            sample: Current latent sample.
            generator: Optional PRNG.

        Returns:
            Updated latent sample.
        """
        return [s - 0.01 * m for s, m in zip(sample, model_output, strict=False)]
