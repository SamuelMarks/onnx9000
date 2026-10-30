"""Tree-to-tensor neural transpilation engine for ONNX9000 (Hummingbird)."""

from onnx9000.optimizer.hummingbird import (
    Strategy,
    TargetHardware,
    TranspilationEngine,
    TreeAbstractions,
    estimate_memory_footprint,
    select_optimal_strategy,
)


class Hummingbird:
    """High-level Hummingbird facade for transpiling tree ensembles into tensor graphs."""

    def __init__(self, target: TargetHardware = TargetHardware.CPU) -> None:
        """Initialize the Hummingbird transpiler with a target hardware profile.

        Args:
            target: Target hardware profile (`CPU`, `CUDA`, `WEBGPU`, `WASM`).
        """
        self.target = target
        self.engine = TranspilationEngine(target=target)

    def process(self, model_name: str) -> str:
        """Process a tree model identifier.

        Args:
            model_name: Name of the tree ensemble model.

        Returns:
            Status summary string.
        """
        return f"Hummingbird processed {model_name} on {self.target.name.lower()}"


def run() -> str:
    """Execute the default Hummingbird module status check.

    Returns:
        Hummingbird status confirmation string.
    """
    return "[hummingbird] processed"


__all__ = [
    "Hummingbird",
    "Strategy",
    "TargetHardware",
    "TranspilationEngine",
    "TreeAbstractions",
    "estimate_memory_footprint",
    "run",
    "select_optimal_strategy",
]
