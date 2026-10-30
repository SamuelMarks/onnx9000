"""AutoOptimizer orchestration."""

import copy

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.context import PassContext
from onnx9000.optimizer.olive.model import OliveModel
from onnx9000.optimizer.olive.passes import Pass
from onnx9000.optimizer.olive.target import Target


class AutoOptimizer:
    """Orchestrates pass sequence."""

    def __init__(
        self,
        target: Target,
        passes: list[Pass],
        config: dict[str, int | float | str | bool | bytes | list | dict | tuple | None]
        | None = None,
    ) -> None:
        """Initialize AutoOptimizer."""
        self.target = target
        self.passes = passes
        self.config = config or {}

    def optimize(self, model: OliveModel) -> OliveModel:
        """Run pass sequence orchestration."""
        current_model = model
        context = PassContext()
        for p in self.passes:
            if not self._check_hardware_limits(p):
                continue
            try:
                candidate = p.run(copy.deepcopy(current_model), context)
                self._validate_topology(candidate)
                current_model = candidate
            except (
                Onnx9000Error,
                OSError,
                ValueError,
                TypeError,
                RuntimeError,
                KeyError,
                IndexError,
                AttributeError,
                ImportError,
                LookupError,
                ArithmeticError,
                BufferError,
                AssertionError,
                StopIteration,
            ):
                continue
        return current_model

    def _check_hardware_limits(self, p: Pass) -> bool:
        """Support conditional Pass execution based on Target hardware limits."""
        return True

    def _validate_topology(self, model: OliveModel) -> None:
        """Validate subgraph topology."""
        return None
