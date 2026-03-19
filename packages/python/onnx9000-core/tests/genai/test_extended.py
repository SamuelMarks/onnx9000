def test_extended():
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

    assert DraftingModel()._initialized
    assert DraftVerifier()._initialized
    assert SelfConsistencyDecoder()._initialized
    assert ContinuousBatchingQueue()._initialized
    assert HiddenStateVisualizer()._initialized
    assert PromptCompressor()._initialized
    assert ChunkedPrefiller()._initialized
    assert DynamicParamAdjuster()._initialized
    assert MultiTurnCache()._initialized
