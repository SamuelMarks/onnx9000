"""Unit tests for GenAI model builder and CLI in `onnx9000.genai.builder`."""

from onnx9000.genai.builder import GenAIBuilder, GenAICLI


def test_genai_builder_and_cli() -> None:
    """Verify GenAIBuilder and GenAICLI methods."""
    assert GenAIBuilder.build("gpt2")["status"] == "built"
    assert GenAIBuilder.export_pytorch(None, None) is None
    assert GenAIBuilder.insert_kv_cache(None) is None
    assert GenAIBuilder.fix_dynamic_axes(None) is None
    assert GenAIBuilder.remove_past_state(None) is None

    assert GenAICLI.run_build("gpt2", "cpu") is None
    assert GenAICLI.run_chat("model.onnx") == "chat_session_started"
