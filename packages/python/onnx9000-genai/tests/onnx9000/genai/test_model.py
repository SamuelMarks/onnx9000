"""Unit tests for GenAI Model in `onnx9000.genai.model`."""

from onnx9000.genai.model import Model
from onnx9000.genai.types import GeneratorParams, ModelParams


def test_genai_model() -> None:
    """Verify Model instantiation and create_generator."""
    mp = ModelParams(
        max_sequence_length=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_size=64,
        vocab_size=100,
        eos_token_id=2,
    )
    model = Model(mp)
    gen = model.create_generator(GeneratorParams(max_length=32))
    assert gen is not None
