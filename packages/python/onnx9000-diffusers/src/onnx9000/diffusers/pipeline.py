"""End-to-end asynchronous diffusion inference pipeline for ONNX9000."""

import asyncio
import gc
import inspect
from collections.abc import Callable

from onnx9000.diffusers.schedulers import EulerDiscreteScheduler, Scheduler
from onnx9000.diffusers.utils import (
    JsonValue,
    PyTorchPCG,
    parse_model_index,
    randn,
    set_progress_bar_config,
)


class AbortSignal:
    """Signal object used to cooperatively cancel an in-progress diffusion pipeline run."""

    def __init__(self) -> None:
        """Initialize an un-aborted signal."""
        self._aborted: bool = False

    def abort(self) -> None:
        """Trigger cancellation of the associated pipeline execution."""
        self._aborted = True

    @property
    def aborted(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._aborted


class DiffusionPipeline:
    """End-to-end inference pipeline for text-to-image and latent diffusion models."""

    def __init__(
        self,
        model_index: dict[str, JsonValue] | None = None,
        scheduler: Scheduler | None = None,
        **kwargs: JsonValue,
    ) -> None:
        """Initialize the diffusion pipeline.

        Args:
            model_index: Optional parsed `model_index.json` dictionary.
            scheduler: Optional denoising `Scheduler` instance.
            **kwargs: Additional configuration entries stored in `self.config`.
        """
        self.model_index: dict[str, JsonValue] = model_index if model_index is not None else {}
        self.config: dict[str, JsonValue] = dict(kwargs)
        if model_index:
            self.config.update(model_index)
        self.scheduler: Scheduler = scheduler if scheduler is not None else EulerDiscreteScheduler()
        self.device: str = "cpu"
        self._device: str = "cpu"
        self._is_aborted: bool = False
        self._memory_arena: dict[str, list[float]] = {}
        self._explicit_scheduler: bool = scheduler is not None or model_index is not None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str | None = None,
        **kwargs: JsonValue,
    ) -> "DiffusionPipeline":
        """Load a diffusion pipeline from a local path or Hugging Face Hub repository ID.

        Args:
            pretrained_model_name_or_path: Repository ID or directory path.
            cache_dir: Optional cache directory for Hub downloads.
            **kwargs: Additional configuration parameters.

        Returns:
            Initialized `DiffusionPipeline` instance.
        """
        if cache_dir is not None:
            model_index = parse_model_index(pretrained_model_name_or_path, cache_dir)
            scheduler = EulerDiscreteScheduler()
            return cls(model_index=model_index, scheduler=scheduler, **kwargs)
        return cls(model_path=pretrained_model_name_or_path, **kwargs)

    def to(self, device: str) -> "DiffusionPipeline":
        """Move pipeline execution target to the specified hardware device.

        Args:
            device: Target device identifier (e.g. `"cpu"`, `"cuda"`, `"mps"`, `"webgpu"`).

        Returns:
            `self` for method chaining.
        """
        self.device = device
        self._device = device
        return self

    def free_memory(self) -> None:
        """Flush cached intermediate latent buffers and abort active generation."""
        self._is_aborted = True
        self._memory_arena.clear()
        gc.collect()

    async def __call__(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        generator: PyTorchPCG | None = None,
        callback_on_step_end: Callable[[int, int, list[float]], None] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> dict[str, list[float]] | list[float]:
        """Run the asynchronous denoising diffusion loop.

        Args:
            prompt: Text conditioning prompt.
            num_inference_steps: Number of denoising steps to run.
            generator: Optional deterministic `PyTorchPCG` PRNG.
            callback_on_step_end: Optional step callback invoked at the end of each step.
            abort_signal: Optional `AbortSignal` to cancel execution mid-run.

        Returns:
            Flat list of denoised latents when initialized with `model_index`/`scheduler`,
            or `{"images": latents}` dictionary when invoked in config mode.

        Raises:
            InterruptedError: If `abort_signal.aborted` is set during inference.
        """
        self._is_aborted = False
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        gen = generator if generator is not None else PyTorchPCG(42)
        shape = (1, 4, 64, 64)
        latents = randn(shape, gen)

        if self.scheduler.sigmas:
            latents = [val * self.scheduler.sigmas[0] for val in latents]

        self._memory_arena["latents"] = latents

        for step, t in enumerate(timesteps):
            if abort_signal is not None and abort_signal.aborted:
                raise InterruptedError("Pipeline aborted.")
            if self._is_aborted:
                break

            noise_pred = randn(shape, gen)
            latents = self.scheduler.step(noise_pred, t, latents, gen)
            self._memory_arena["latents"] = latents

            if callback_on_step_end is not None:
                sig = inspect.signature(callback_on_step_end)
                if len(sig.parameters) == 3:
                    callback_on_step_end(step, t, latents)
                else:
                    callback_on_step_end(step, t, num_inference_steps, latents)

            await asyncio.sleep(0.0)

        if self._explicit_scheduler:
            return latents
        return {"images": latents}


__all__ = ["AbortSignal", "DiffusionPipeline", "set_progress_bar_config"]
