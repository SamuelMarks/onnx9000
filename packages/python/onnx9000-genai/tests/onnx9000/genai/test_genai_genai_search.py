"""Unit tests for search algorithms in `onnx9000.genai.search`."""

import struct

from onnx9000.core.ir import Tensor
from onnx9000.genai.search import (
    BeamSearchAlgorithm,
    BeamSearchState,
    GreedySearch,
    MultinomialSampling,
    SearchAlgorithm,
)


def test_search_algorithms() -> None:
    """Verify SearchAlgorithm, GreedySearch, MultinomialSampling, BeamSearchState, and BeamSearchAlgorithm."""
    buf = bytearray(struct.pack("<3f", 1.0, 9.0, 2.0))
    logits = Tensor(name="logits", shape=(1, 3), data=buf)

    assert SearchAlgorithm().select_next_token(logits, []) == 0
    assert GreedySearch().select_next_token(logits, []) == 1
    assert MultinomialSampling(seed=123).select_next_token(logits, []) in (0, 1, 2)

    state = BeamSearchState(num_beams=2, num_return_sequences=1)
    state.add_finished(5.0, [1, 2])
    assert state.get_best_finished()[0][1] == [1, 2]

    beam_algo = BeamSearchAlgorithm(state)
    top = beam_algo.process_logits(logits, 0)
    assert len(top) == 2
    assert top[0][1] == 1
