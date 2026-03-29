"""Module providing functionality for test_extended."""

"""Test extended."""


def test_extended():
    """Provides functional implementation."""
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

    assert DraftingModel()._initialized
    assert DraftVerifier()._initialized
    assert SelfConsistencyDecoder()._initialized
    assert ContinuousBatchingQueue()._initialized
    assert HiddenStateVisualizer()._initialized
    assert PromptCompressor()._initialized
    assert ChunkedPrefiller()._initialized
    assert DynamicParamAdjuster()._initialized
    assert MultiTurnCache()._initialized
