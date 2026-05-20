import pytest
from onnx9000_llama_web import run_model


def test_run_model():
    assert run_model("llama") == "[LLaMA-Web] processing llama"


def test_run_model_invalid():
    with pytest.raises(ValueError):
        run_model("")
