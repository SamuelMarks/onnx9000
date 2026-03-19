import struct
from typing import List, Tuple

from ..core.ir import Tensor


class SearchAlgorithm:
    """Base class for search and sampling algorithms."""

    def select_next_token(self, logits: Tensor, input_ids: list[int]) -> int:
        raise NotImplementedError()


class GreedySearch(SearchAlgorithm):
    """Selects the token with the highest probability."""

    def select_next_token(self, logits: Tensor, input_ids: list[int]) -> int:
        if logits.data is None:
            return 0

        vocab_size = logits.shape[-1] if logits.shape else 1
        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4

        offset = len(logits.data) - vocab_size * itemsize
        if offset < 0:
            return 0

        max_val = float("-inf")
        max_idx = 0

        for i in range(vocab_size):
            start = offset + i * itemsize
            if start >= 0 and start + itemsize <= len(logits.data):
                val = struct.unpack("<f", logits.data[start : start + itemsize])[0]
                import math

                if math.isnan(val) or math.isinf(val) and val > 0:
                    return 0
                if val > max_val:
                    max_val = val
                    max_idx = i

        return max_idx


class MultinomialSampling(SearchAlgorithm):
    """Samples the next token from the multinomial distribution given by logits."""

    def __init__(self, seed: int = None):
        import random

        self.rng = random.Random(seed) if seed is not None else random

    def select_next_token(self, logits: Tensor, input_ids: list[int]) -> int:
        if logits.data is None:
            return 0

        import math
        import struct

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize

        offset = len(logits.data) - vocab_size * itemsize
        if offset < 0:
            return 0

        vals = []
        for i in range(vocab_size):
            start = offset + i * itemsize
            if start >= 0 and start + itemsize <= len(logits.data):
                val = struct.unpack("<f", logits.data[start : start + itemsize])[0]
                vals.append(val)

        max_val = max(vals) if vals else 0
        exp_vals = [math.exp(v - max_val) for v in vals]
        sum_exp = sum(exp_vals)
        probs = [e / sum_exp for e in exp_vals]

        r = self.rng.random()
        cumulative = 0.0
        for i, prob in enumerate(probs):
            cumulative += prob
            if r <= cumulative:
                return i

        return len(vals) - 1 if vals else 0


class BeamSearchState:
    """Manages beam search state (beam scores, beam tokens, beam histories)."""

    def __init__(self, num_beams: int, num_return_sequences: int):
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.active_beams = []  # list of tuples (score, [tokens])
        self.finished_beams = []  # list of tuples (score, [tokens])

    def add_finished(self, score: float, tokens: list[int]):
        self.finished_beams.append((score, tokens))

    def get_best_finished(self):
        # Sort by score descending
        self.finished_beams.sort(key=lambda x: x[0], reverse=True)
        return self.finished_beams[: self.num_return_sequences]


class BeamSearchAlgorithm(SearchAlgorithm):
    """Implements Beam Search algorithm, pruning, and sorting."""

    def __init__(self, state: BeamSearchState):
        self.state = state

    def process_logits(self, next_token_logits: Tensor, beam_idx: int) -> list[tuple[float, int]]:
        # Mock logic to extract top num_beams from a single beam's logits
        import struct

        itemsize = (
            next_token_logits.dtype.itemsize if hasattr(next_token_logits.dtype, "itemsize") else 4
        )
        vocab_size = (
            next_token_logits.shape[-1]
            if next_token_logits.shape
            else len(next_token_logits.data) // itemsize
        )

        offset = len(next_token_logits.data) - vocab_size * itemsize
        vals = []
        for i in range(vocab_size):
            start = offset + i * itemsize
            if start + itemsize <= len(next_token_logits.data):
                val = struct.unpack("<f", next_token_logits.data[start : start + itemsize])[0]
                vals.append((val, i))

        vals.sort(key=lambda x: x[0], reverse=True)
        return vals[: self.state.num_beams]

    def prune_and_sort_beams(self, candidates: list[tuple[float, list[int]]]):
        """Implements beam search pruning and sorting."""
        candidates.sort(key=lambda x: x[0], reverse=True)
        self.state.active_beams = candidates[: self.state.num_beams]
