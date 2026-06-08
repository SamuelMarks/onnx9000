import pytest
from onnx9000_llama_web import run_model


def test_run_model():
    with pytest.raises(ValueError):
        run_model("")
    assert run_model("test") == "[LLaMA-Web] processing test"
