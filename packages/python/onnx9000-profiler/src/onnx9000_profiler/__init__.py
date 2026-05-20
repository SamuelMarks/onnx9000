"""Profiler package."""


class Profiler:
    """A simple ONNX model profiler."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.peak_memory = 0.0

    def run(self):
        """Run the profiler."""
        # Simulated run
        self.peak_memory = 42.5

    def get_peak_memory(self) -> float:
        """Get the simulated peak memory in MB."""
        return self.peak_memory
