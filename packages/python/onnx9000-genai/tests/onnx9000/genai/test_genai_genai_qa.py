"""Unit tests for GenAI QA and debugging tools in `onnx9000.genai.qa`."""

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


def test_qa_tools() -> None:
    """Verify StepDebuggerUI, AttentionMapVisualizer, BeamSearchTreeVisualizer, SamplingConfigLinter, ChromeTracer, BrokenModelSuite, HardwareBugDatabase, TokenizerEdgeCasesTester, LogitComparer, and FeatureToggles."""
    ui = StepDebuggerUI()
    ui.record_step("s1", {"loss": 0.1})
    assert len(ui.history) == 1

    amv = AttentionMapVisualizer()
    amv.add_map([[1.0]])
    assert "div" in amv.generate_html()

    bst = BeamSearchTreeVisualizer()
    bst.add_node("r", "c", 0.5)
    assert bst.export_json()["c"]["score"] == 0.5

    linter = SamplingConfigLinter()
    assert len(linter.lint({"temperature": -1.0})) == 1

    tracer = ChromeTracer()
    tracer.log_event("prefill", 1.0)
    assert len(tracer.events) == 1

    bms = BrokenModelSuite()
    bms.register("broken_1")
    assert bms.is_broken("broken_1") is True

    hbd = HardwareBugDatabase()
    hbd.add_bug("gpu0", "nan_issue")
    assert hbd.get_bugs("gpu0") == "nan_issue"

    tet = TokenizerEdgeCasesTester()
    tet.add_case("emoji")
    assert tet.run_tests(None) is True

    lc = LogitComparer()
    lc.set_baseline([1.0, 2.0])
    assert lc.compare([1.0, 2.0]) == 0.0

    ft = FeatureToggles()
    ft.enable("flash_attn")
    assert ft.is_enabled("flash_attn") is True
