def test_qa():
    from onnx9000.genai.qa import (
        StepDebuggerUI,
        AttentionMapVisualizer,
        BeamSearchTreeVisualizer,
        SamplingConfigLinter,
        ChromeTracer,
        BrokenModelSuite,
        HardwareBugDatabase,
        TokenizerEdgeCasesTester,
        LogitComparer,
        FeatureToggles,
    )

    assert StepDebuggerUI()._initialized
    assert AttentionMapVisualizer()._initialized
    assert BeamSearchTreeVisualizer()._initialized
    assert SamplingConfigLinter()._initialized
    assert ChromeTracer()._initialized
    assert BrokenModelSuite()._initialized
    assert HardwareBugDatabase()._initialized
    assert TokenizerEdgeCasesTester()._initialized
    assert LogitComparer()._initialized
    assert FeatureToggles()._initialized
