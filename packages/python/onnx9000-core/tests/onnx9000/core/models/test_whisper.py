"""Unit tests for Whisper encoder/decoder model builder in `onnx9000.core.models.whisper`."""

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.whisper import (
    Whisper,
    WhisperDecoder,
    WhisperDecoderLayer,
    WhisperEncoder,
    WhisperEncoderLayer,
    get_param,
    whisper_tiny,
)


def test_whisper_components_and_tiny() -> None:
    """Verify WhisperEncoderLayer, WhisperEncoder, WhisperDecoderLayer, WhisperDecoder, Whisper, and whisper_tiny."""
    p = get_param("w", [16, 16])
    assert isinstance(p, Variable)
    assert p.shape == [16, 16]

    enc_layer = WhisperEncoderLayer(
        d_model=64, encoder_attention_heads=4, encoder_ffn_dim=128, prefix="enc.0"
    )
    x_enc = Tensor("x_enc", (1, 10, 64), 1)
    assert isinstance(enc_layer(x_enc), Tensor)

    enc = WhisperEncoder(
        d_model=64, encoder_attention_heads=4, encoder_ffn_dim=128, encoder_layers=1
    )
    mel = Tensor("mel", (1, 80, 100), 1)
    hidden = enc(mel)
    assert isinstance(hidden, Tensor)

    dec_layer = WhisperDecoderLayer(
        d_model=64, decoder_attention_heads=4, decoder_ffn_dim=128, prefix="dec.0"
    )
    x_dec = Tensor("x_dec", (1, 5, 64), 1)
    assert isinstance(dec_layer(x_dec, hidden), Tensor)

    dec = WhisperDecoder(
        vocab_size=100, d_model=64, decoder_attention_heads=4, decoder_ffn_dim=128, decoder_layers=1
    )
    ids = Tensor("ids", (1, 5), 7)
    assert isinstance(dec(ids, hidden), Tensor)

    model = whisper_tiny(vocab_size=100)
    assert isinstance(model, Whisper)
    out = model(mel, ids)
    assert isinstance(out, Tensor)
