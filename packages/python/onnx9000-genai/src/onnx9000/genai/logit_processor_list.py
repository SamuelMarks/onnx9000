"""Implementation of logit processor list (onnx9000 / genai / logit_processor_list.py) providing LogitProcessorList."""

from ..core.ir import Tensor
from .logit_processors import LogitProcessor


class LogitProcessorList(LogitProcessor):
    """Pipeline for multiple LogitProcessors."""

    def __init__(self, processors: list[LogitProcessor] | None = None) -> None:
        """Initialize LogitProcessorList with configured parameters and state.

        Args:
            processors: Parameter specifying processors (list[LogitProcessor]).
        """
        self.processors = processors or []

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        for processor in self.processors:
            logits = processor(input_ids, logits)
        return logits
