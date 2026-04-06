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


def test_step_debugger():
    debugger = StepDebuggerUI()
    debugger.record_step("step1", {"x": 1})
    assert len(debugger.history) == 1
    debugger.render()


def test_attention_map():
    viz = AttentionMapVisualizer()
    viz.add_map([[0.5, 0.5]])
    assert "1" in viz.generate_html()


def test_beam_search():
    viz = BeamSearchTreeVisualizer()
    viz.add_node("root", "child1", 0.9)
    assert viz.export_json()["child1"]["score"] == 0.9


def test_linter():
    linter = SamplingConfigLinter()
    errors = linter.lint({"temperature": -1.0})
    assert len(errors) == 1
    assert "Temperature" in errors[0]


def test_chrome_tracer():
    tracer = ChromeTracer()
    tracer.log_event("start", 123.4)
    assert len(tracer.events) == 1


def test_broken_model():
    suite = BrokenModelSuite()
    suite.register("bad_model")
    assert suite.is_broken("bad_model")
    assert not suite.is_broken("good_model")


def test_hardware_bug():
    db = HardwareBugDatabase()
    db.add_bug("device_x", "bug1")
    assert db.get_bugs("device_x") == "bug1"
    assert db.get_bugs("device_y") is None


def test_tokenizer_edge_cases():
    tester = TokenizerEdgeCasesTester()
    assert not tester.run_tests(None)
    tester.add_case("text")
    assert tester.run_tests(None)


def test_logit_comparer():
    comp = LogitComparer()
    comp.set_baseline([1.0, 2.0])
    assert comp.compare([1.0, 2.0]) == 0.0


def test_feature_toggles():
    toggles = FeatureToggles()
    toggles.enable("feat1")
    assert toggles.is_enabled("feat1")
    assert not toggles.is_enabled("feat2")
