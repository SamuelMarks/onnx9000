"""Unit tests for extended GenAI decoding utilities in `onnx9000.genai.extended`."""

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


def test_extended_utilities() -> None:
    """Verify speculative drafting, self-consistency, continuous batching, prompt compression, chunked prefill, and multi-turn cache."""
    dm = DraftingModel()
    assert len(dm.draft("prompt")) == 2

    dv = DraftVerifier()
    assert dv.verify("hello world", "hello") is True

    sc = SelfConsistencyDecoder()
    assert sc.decode(["x", "y", "x"]) == "x"

    cb = ContinuousBatchingQueue(2)
    cb.enqueue("r1")
    cb.enqueue("r2")
    assert cb.dequeue_batch() == ["r1", "r2"]

    hsv = HiddenStateVisualizer()
    hsv.record_state([0.1, 0.2])
    assert hsv.states == [[0.1, 0.2]]

    pc = PromptCompressor(0.5)
    assert pc.compress("abcd") == "ab"

    cp = ChunkedPrefiller(2)
    assert cp.prefill([1, 2, 3, 4]) == [[1, 2], [3, 4]]

    dpa = DynamicParamAdjuster()
    dpa.adjust("top_p", 0.9)
    assert dpa.params["top_p"] == 0.9

    mtc = MultiTurnCache()
    mtc.update("s1", "msg1")
    assert mtc.cache["s1"] == ["msg1"]
