"""Provide functionality for this module."""

import math

from ..core.ir import Tensor
from .logit_processors import LogitProcessor


class TopPLogitProcessor(LogitProcessor):
    """Filters logits based on top-P (Nucleus) sampling."""

    def __init__(self, top_p: float):
        """Initialize the instance."""
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("top_p must be strictly positive and <= 1.0")
        self.top_p = top_p

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if self.top_p >= 1.0 or logits.data is None:
            return logits

        import struct

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize

        offset = len(logits.data) - vocab_size * itemsize

        vals = []
        for i in range(vocab_size):
            start = offset + i * itemsize
            if start >= 0 and start + itemsize <= len(logits.data):
                val = struct.unpack("<f", logits.data[start : start + itemsize])[0]
                vals.append((val, i))

        # Softmax
        max_val = max(v[0] for v in vals) if vals else 0
        exp_vals = [math.exp(v[0] - max_val) for v in vals]
        sum_exp = sum(exp_vals)
        probs = [e / sum_exp for e in exp_vals]

        # Sort by probability
        sorted_probs = sorted(zip(probs, [v[1] for v in vals]), key=lambda x: x[0], reverse=True)

        cumulative_prob = 0.0
        threshold_idx = len(sorted_probs) - 1

        for i, (prob, idx) in enumerate(sorted_probs):
            cumulative_prob += prob
            if cumulative_prob > self.top_p:
                threshold_idx = i
                break

        # Indices to mask
        indices_to_remove = [idx for _, idx in sorted_probs[threshold_idx + 1 :]]

        if indices_to_remove:
            new_data = bytearray(logits.data)
            neg_inf = struct.pack("<f", float("-inf"))
            for idx in indices_to_remove:
                start = offset + idx * itemsize
                new_data[start : start + itemsize] = neg_inf
            return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)

        return logits
