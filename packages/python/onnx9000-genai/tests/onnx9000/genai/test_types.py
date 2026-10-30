"""Unit tests for ModelParams and GeneratorParams in `onnx9000.genai.types`."""

from onnx9000.genai.types import GeneratorParams, ModelParams


def test_model_and_generator_params() -> None:
    """Verify ModelParams and GeneratorParams store configuration attributes."""
    mp = ModelParams(
        max_sequence_length=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        hidden_size=768,
        vocab_size=32000,
        eos_token_id=2,
        bos_token_id=1,
        pad_token_id=0,
    )
    assert mp.vocab_size == 32000
    assert mp.bos_token_id == 1

    gp = GeneratorParams(max_length=128, temperature=0.7, top_k=40, top_p=0.95, do_sample=True)
    assert gp.max_length == 128
    assert gp.temperature == 0.7
    assert gp.do_sample is True
