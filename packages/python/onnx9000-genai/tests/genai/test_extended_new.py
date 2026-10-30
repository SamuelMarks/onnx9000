"""Unit tests and verification suite for extended new functionality."""

from onnx9000.genai.extended import (
    ChunkedPrefiller,
    ContinuousBatchingQueue,
    DraftingModel,
    DraftVerifier,
    DynamicParamAdjuster,
    HiddenStateVisualizer,
    MultiTurnCache,
    PromptCompressor,
    SelfConsistencyDecoder,
)


def test_drafting_model() -> None:
    """Verify that drafting model behaves as expected."""
    model = DraftingModel()
    res = model.draft("hello")
    assert len(res) == 2
    assert "draft1" in res[0]


def test_draft_verifier() -> None:
    """Verify that draft verifier behaves as expected."""
    verifier = DraftVerifier()
    assert verifier.verify("test draft", "test")
    assert not verifier.verify("other", "test")
    assert verifier.verified_count == 2


def test_self_consistency() -> None:
    """Verify that self consistency behaves as expected."""
    decoder = SelfConsistencyDecoder()
    assert decoder.decode([]) == ""
    assert decoder.decode(["a", "b", "a"]) == "a"


def test_continuous_queue() -> None:
    """Verify that continuous queue behaves as expected."""
    q = ContinuousBatchingQueue(2)
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    assert q.dequeue_batch() == [1, 2]
    assert q.queue == [3]


def test_hidden_state() -> None:
    """Verify that hidden state behaves as expected."""
    vis = HiddenStateVisualizer()
    vis.record_state([1.0, 2.0])
    assert vis.states == [[1.0, 2.0]]


def test_compressor() -> None:
    """Verify that compressor behaves as expected."""
    comp = PromptCompressor(0.5)
    assert comp.compress("1234") == "12"


def test_chunked_prefiller() -> None:
    """Verify that chunked prefiller behaves as expected."""
    pref = ChunkedPrefiller(2)
    assert pref.prefill([1, 2, 3]) == [[1, 2], [3]]


def test_dynamic_param() -> None:
    """Verify that dynamic param behaves as expected."""
    adj = DynamicParamAdjuster()
    adj.adjust("temp", 0.5)
    assert adj.params["temp"] == 0.5


def test_multi_turn_cache() -> None:
    """Verify that multi turn cache behaves as expected."""
    cache = MultiTurnCache()
    cache.update("sess1", "data")
    assert cache.cache["sess1"] == ["data"]
