import math
from typing import List


class Scheduler:
    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps: int = num_train_timesteps
        self.timesteps: List[int] = list(range(num_train_timesteps - 1, -1, -1))


class DDIMScheduler(Scheduler):
    def step(self, model_output: List[float], timestep: int, sample: List[float]) -> List[float]:
        """Implement DDIMScheduler natively in JS/Python."""
        alpha_prod_t = 1.0 - (timestep / self.num_train_timesteps)
        beta_prod_t = 1 - alpha_prod_t
        return [
            (s - math.sqrt(beta_prod_t) * o) / math.sqrt(alpha_prod_t)
            for s, o in zip(sample, model_output)
        ]


class DDPMScheduler(Scheduler):
    def step(self, model_output: List[float], timestep: int, sample: List[float]) -> List[float]:
        """Implement DDPMScheduler."""
        alpha_t = 1.0 - (timestep / self.num_train_timesteps)
        return [(s - (1 - alpha_t) * o) / math.sqrt(alpha_t) for s, o in zip(sample, model_output)]


class EulerDiscreteScheduler(Scheduler):
    def step(self, model_output: List[float], timestep: int, sample: List[float]) -> List[float]:
        """Implement EulerDiscreteScheduler."""
        sigma = timestep / self.num_train_timesteps
        return [s + o * sigma for s, o in zip(sample, model_output)]


class LCMScheduler(Scheduler):
    def step(self, model_output: List[float], timestep: int, sample: List[float]) -> List[float]:
        """Implement LCMScheduler (Latent Consistency Models) for 2-4 step generation."""
        return [s - o for s, o in zip(sample, model_output)]
