"""Module providing core logic and structural definitions."""

import math
from typing import List, Callable


def greedy_search(logits: List[List[float]], eos_token_id: int = -1) -> List[int]:
    """Provides semantic functionality and verification."""
    out = []
    for step_logits in logits:
        best_idx = 0
        best_score = float("-inf")
        for j, val in enumerate(step_logits):
            if val > best_score:
                best_score = val
                best_idx = j
        out.append(best_idx)
        if best_idx == eos_token_id:
            break
    return out


def sample_top_k_top_p(
    logits: List[float], top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0
) -> int:
    """Provides semantic functionality and verification."""
    import random

    if temperature != 1.0 and temperature > 0:
        logits = [l / temperature for l in logits]

    max_logit = max(logits)
    probs = [math.exp(l - max_logit) for l in logits]
    sum_probs = sum(probs)
    probs = [p / sum_probs for p in probs]

    indexed_probs = [(p, i) for i, p in enumerate(probs)]
    indexed_probs.sort(key=lambda x: x[0], reverse=True)

    filtered = indexed_probs

    if top_k > 0 and top_k < len(filtered):
        filtered = filtered[:top_k]

    if top_p < 1.0:
        cum_sum = 0.0
        cutoff_idx = len(filtered)
        for i, item in enumerate(filtered):
            cum_sum += item[0]
            if cum_sum >= top_p:
                cutoff_idx = i + 1
                break
        filtered = filtered[:cutoff_idx]

    new_sum = sum(x[0] for x in filtered)
    filtered = [(x[0] / new_sum, x[1]) for x in filtered]

    rand = random.random()
    acc = 0.0
    for p, i in filtered:
        acc += p
        if rand <= acc:
            return i

    return filtered[-1][1]


class Beam:
    """Provides semantic functionality and verification."""

    def __init__(self, seq: List[int], score: float, is_finished: bool):
        """Provides semantic functionality and verification."""
        self.seq = seq
        self.score = score
        self.is_finished = is_finished


def beam_search(
    logits_fn: Callable[[List[List[int]]], List[List[float]]],
    initial_input: List[int],
    beam_size: int,
    max_steps: int,
    eos_token_id: int = -1,
) -> List[int]:
    """Provides semantic functionality and verification."""
    beams = [Beam(list(initial_input), 0.0, False)]

    for _ in range(max_steps):
        active_beams = [b for b in beams if not b.is_finished]
        finished_beams = [b for b in beams if b.is_finished]

        if not active_beams:
            break

        seqs = [b.seq for b in active_beams]
        next_logits_batch = logits_fn(seqs)

        new_beams = []
        for b_idx, beam in enumerate(active_beams):
            logits = next_logits_batch[b_idx]

            max_l = max(logits)
            log_probs = [math.exp(l - max_l) for l in logits]
            sum_p = sum(log_probs)
            log_probs = [math.log(p / sum_p) for p in log_probs]

            indexed = [(lp, i) for i, lp in enumerate(log_probs)]
            indexed.sort(key=lambda x: x[0], reverse=True)

            for i in range(min(beam_size, len(indexed))):
                next_token = indexed[i][1]
                next_score = beam.score + indexed[i][0]
                new_beams.append(
                    Beam(
                        seq=beam.seq + [next_token],
                        score=next_score,
                        is_finished=(next_token == eos_token_id),
                    )
                )

        combined = finished_beams + new_beams
        combined.sort(key=lambda b: b.score, reverse=True)
        beams = combined[:beam_size]

    return beams[0].seq
