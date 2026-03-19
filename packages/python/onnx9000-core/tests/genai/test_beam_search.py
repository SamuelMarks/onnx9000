def test_beam_search_logic():
    """Test beam search logic end-to-end with GeneratorParams and basic state."""
    from onnx9000.core.ir import Graph, Tensor
    from onnx9000.genai.search import BeamSearchState
    from onnx9000.genai.state import KVCache, State
    from onnx9000.genai.types import GeneratorParams

    params = GeneratorParams(max_length=10, num_beams=2, num_return_sequences=1)

    # set up basic graph and state
    graph = Graph("mock_graph")
    cache = KVCache()
    State(graph, cache)

    # set up beam search state
    beam_state = BeamSearchState(
        num_beams=params.num_beams, num_return_sequences=params.num_return_sequences
    )

    # Assert initial state
    assert len(beam_state.active_beams) == 0
    assert len(beam_state.finished_beams) == 0

    # Add finished beams and assert
    beam_state.add_finished(score=10.0, tokens=[1, 2, 3])
    beam_state.add_finished(score=12.0, tokens=[4, 5, 6])

    best = beam_state.get_best_finished()
    assert len(best) == params.num_return_sequences
    assert best[0][0] == 12.0
    assert best[0][1] == [4, 5, 6]
