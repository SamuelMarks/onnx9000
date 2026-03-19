import pytest
import struct
from onnx9000.core.ir import Tensor
from onnx9000.genai.search import GreedySearch, MultinomialSampling


def create_logits(vals):
    data = bytearray(len(vals) * 4)
    for i, v in enumerate(vals):
        data[i * 4 : (i + 1) * 4] = struct.pack("<f", v)
    return Tensor(
        name="logits",
        shape=(1, len(vals)),
        data=data,
        dtype=type("mock", (), {"itemsize": 4}),
    )


def test_greedy_search():
    search = GreedySearch()
    logits = create_logits([1.0, 5.0, 3.0])
    idx = search.select_next_token(logits, [])
    assert idx == 1


def test_multinomial_sampling():
    search = MultinomialSampling(seed=42)
    logits = create_logits([1.0, 5.0, 3.0])
    idx = search.select_next_token(logits, [])
    assert idx in [0, 1, 2]


def test_greedy_search_edge_cases():
    from onnx9000.genai.search import GreedySearch, SearchAlgorithm
    from onnx9000.core.ir import Tensor

    # base not implemented
    sa = SearchAlgorithm()
    with pytest.raises(NotImplementedError):
        sa.select_next_token(None, [])

    gs = GreedySearch()
    # data is none
    t_none = Tensor(name="x", shape=(1, 10), data=None)
    assert gs.select_next_token(t_none, []) == 0

    # offset negative
    t_neg = Tensor(
        name="x",
        shape=(1, 10),
        data=bytearray(4),
        dtype=type("mock", (), {"itemsize": 4}),
    )
    assert gs.select_next_token(t_neg, []) == 0


def test_multinomial_edge_cases():
    from onnx9000.genai.search import MultinomialSampling
    from onnx9000.core.ir import Tensor

    ms = MultinomialSampling(seed=None)
    t_none = Tensor(name="x", shape=(1, 10), data=None)
    assert ms.select_next_token(t_none, []) == 0

    t_neg = Tensor(
        name="x",
        shape=(1, 10),
        data=bytearray(4),
        dtype=type("mock", (), {"itemsize": 4}),
    )
    assert ms.select_next_token(t_neg, []) == 0

    # empty vals
    t_empty = Tensor(
        name="x",
        shape=(1, 0),
        data=bytearray(0),
        dtype=type("mock", (), {"itemsize": 4}),
    )
    assert ms.select_next_token(t_empty, []) == 0


def test_beam_search():
    from onnx9000.genai.search import BeamSearchState, BeamSearchAlgorithm
    from onnx9000.core.ir import Tensor
    import struct

    state = BeamSearchState(num_beams=2, num_return_sequences=1)
    state.add_finished(10.0, [1, 2])
    state.add_finished(5.0, [3, 4])

    best = state.get_best_finished()
    assert len(best) == 1
    assert best[0][0] == 10.0

    algo = BeamSearchAlgorithm(state)
    with pytest.raises(NotImplementedError):
        algo.select_next_token(None, [])

    data = bytearray(12)
    for i, v in enumerate([1.0, 5.0, 3.0]):
        data[i * 4 : (i + 1) * 4] = struct.pack("<f", v)

    logits = Tensor(name="x", shape=(1, 3), data=data, dtype=type("mock", (), {"itemsize": 4}))
    top = algo.process_logits(logits, 0)
    assert len(top) == 2
    assert top[0][1] == 1  # score 5.0
    assert top[1][1] == 2  # score 3.0

    algo.prune_and_sort_beams([(1.0, [1]), (5.0, [2]), (3.0, [3])])
    assert len(state.active_beams) == 2
    assert state.active_beams[0][0] == 5.0
    assert state.active_beams[1][0] == 3.0


def test_search_missing():
    from onnx9000.genai.search import GreedySearch, MultinomialSampling
    from onnx9000.core.ir import Tensor

    t_short = Tensor(
        name="x",
        shape=(1, 2),
        data=bytearray(4),
        dtype=type("mock", (), {"itemsize": 4}),
    )

    gs = GreedySearch()
    gs.select_next_token(t_short, [])

    ms = MultinomialSampling()
    ms.select_next_token(t_short, [])


def test_search_38():
    from onnx9000.genai.search import GreedySearch
    from onnx9000.core.ir import Tensor
    import struct
    import math

    search = GreedySearch()
    data = bytearray(8)
    data[0:4] = struct.pack("<f", float("nan"))
    data[4:8] = struct.pack("<f", float("inf"))
    t = Tensor(name="", shape=(2,), data=data, dtype=type("m", (), {"itemsize": 4}))
    assert search.select_next_token(t, []) == 0
