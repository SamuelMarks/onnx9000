import pytest
from onnx9000.genai.extended import (
    DraftingModel,
    DraftVerifier,
    SelfConsistencyDecoder,
    ContinuousBatchingQueue,
    HiddenStateVisualizer,
    PromptCompressor,
    ChunkedPrefiller,
    DynamicParamAdjuster,
    MultiTurnCache,
)


def test_drafting_model():
    model = DraftingModel()
    res = model.draft("hello")
    assert len(res) == 2
    assert "draft1" in res[0]


def test_draft_verifier():
    verifier = DraftVerifier()
    assert verifier.verify("test draft", "test")
    assert not verifier.verify("other", "test")
    assert verifier.verified_count == 2


def test_self_consistency():
    decoder = SelfConsistencyDecoder()
    assert decoder.decode([]) == ""
    assert decoder.decode(["a", "b", "a"]) == "a"


def test_continuous_queue():
    q = ContinuousBatchingQueue(2)
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    assert q.dequeue_batch() == [1, 2]
    assert q.queue == [3]


def test_hidden_state():
    vis = HiddenStateVisualizer()
    vis.record_state([1.0, 2.0])
    assert vis.states == [[1.0, 2.0]]


def test_compressor():
    comp = PromptCompressor(0.5)
    assert comp.compress("1234") == "12"


def test_chunked_prefiller():
    pref = ChunkedPrefiller(2)
    assert pref.prefill([1, 2, 3]) == [[1, 2], [3]]


def test_dynamic_param():
    adj = DynamicParamAdjuster()
    adj.adjust("temp", 0.5)
    assert adj.params["temp"] == 0.5


def test_multi_turn_cache():
    cache = MultiTurnCache()
    cache.update("sess1", "data")
    assert cache.cache["sess1"] == ["data"]
