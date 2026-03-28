"""Module providing functionality for test_qa."""

"""Test qa."""


def test_qa():
    """Docstring."""
    from onnx9000.genai.qa import (
        AttentionMapVisualizer,
        BeamSearchTreeVisualizer,
        BrokenModelSuite,
        ChromeTracer,
        FeatureToggles,
        HardwareBugDatabase,
        LogitComparer,
        SamplingConfigLinter,
        StepDebuggerUI,
        TokenizerEdgeCasesTester,
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
