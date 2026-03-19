"""Provide functionality for this module."""

from ..core.ir import Tensor
from .logit_processors import LogitProcessor


class LogitProcessorList(LogitProcessor):
    """Pipeline for multiple LogitProcessors."""

    def __init__(self, processors: list[LogitProcessor] = None):
        """Initialize the instance."""
        self.processors = processors or []

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        for processor in self.processors:
            logits = processor(input_ids, logits)
        return logits
