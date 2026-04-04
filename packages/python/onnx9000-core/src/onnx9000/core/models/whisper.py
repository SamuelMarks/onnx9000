"""Module docstring."""

from typing import Any, Optional

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.ops import add
from onnx9000.core.primitives import ConvND, Gelu, Gemm, LayerNormalization, MultiHeadAttention


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:  # noqa: D103
    return Variable(name=name, shape=shape, dtype=dtype)


class WhisperEncoderLayer:  # noqa: D101
    def __init__(  # noqa: D107
        self, d_model: int, encoder_attention_heads: int, encoder_ffn_dim: int, prefix: str = ""
    ):
        self.prefix = prefix
        self.d_model = d_model

        self.self_attn = MultiHeadAttention(num_heads=encoder_attention_heads)
        self.self_attn_layer_norm = LayerNormalization((d_model,))

        self.fc1 = Gemm(trans_b=1)
        self.act = Gelu()
        self.fc2 = Gemm(trans_b=1)
        self.final_layer_norm = LayerNormalization((d_model,))

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        identity = x
        x_norm = self.self_attn_layer_norm(
            x,
            get_param(f"{self.prefix}.self_attn_layer_norm.weight", [self.d_model]),
            get_param(f"{self.prefix}.self_attn_layer_norm.bias", [self.d_model]),
        )
        x_attn = self.self_attn(x_norm, x_norm, x_norm)
        x = add(identity, x_attn)

        identity = x
        x_norm = self.final_layer_norm(
            x,
            get_param(f"{self.prefix}.final_layer_norm.weight", [self.d_model]),
            get_param(f"{self.prefix}.final_layer_norm.bias", [self.d_model]),
        )
        x_ffn = self.fc1(
            x_norm,
            get_param(f"{self.prefix}.fc1.weight", [self.d_model * 4, self.d_model]),
            get_param(f"{self.prefix}.fc1.bias", [self.d_model * 4]),
        )
        x_ffn = self.act(x_ffn)
        x_ffn = self.fc2(
            x_ffn,
            get_param(f"{self.prefix}.fc2.weight", [self.d_model, self.d_model * 4]),
            get_param(f"{self.prefix}.fc2.bias", [self.d_model]),
        )

        return add(identity, x_ffn)


class WhisperEncoder:  # noqa: D101
    def __init__(  # noqa: D107
        self,
        d_model: int = 512,
        encoder_attention_heads: int = 8,
        encoder_ffn_dim: int = 2048,
        encoder_layers: int = 6,
    ):
        self.d_model = d_model
        self.conv1 = ConvND(1, 80, d_model, kernel_size=3, padding=1)
        self.act1 = Gelu()
        self.conv2 = ConvND(1, d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.act2 = Gelu()

        self.layers = [
            WhisperEncoderLayer(
                d_model, encoder_attention_heads, encoder_ffn_dim, prefix=f"layers.{i}"
            )
            for i in range(encoder_layers)
        ]
        self.layer_norm = LayerNormalization((d_model,))

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        # Input x: [batch, 80, seq_len]
        x = self.conv1(
            x,
            get_param("conv1.weight", [self.d_model, 80, 3]),
            get_param("conv1.bias", [self.d_model]),
        )
        x = self.act1(x)
        x = self.conv2(
            x,
            get_param("conv2.weight", [self.d_model, self.d_model, 3]),
            get_param("conv2.bias", [self.d_model]),
        )
        x = self.act2(x)

        # Transpose to [batch, seq_len, d_model] for transformer
        from onnx9000.core.ops import transpose

        x = transpose(x, perm=[0, 2, 1])

        pos_embed = get_param("embed_positions.weight", [1, 1500, self.d_model])
        # Add pos_embed conceptually (needs slice in real forward based on shape)
        x = add(x, pos_embed)

        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(
            x,
            get_param("layer_norm.weight", [self.d_model]),
            get_param("layer_norm.bias", [self.d_model]),
        )
        return x


class WhisperDecoderLayer:  # noqa: D101
    def __init__(  # noqa: D107
        self, d_model: int, decoder_attention_heads: int, decoder_ffn_dim: int, prefix: str = ""
    ):
        self.prefix = prefix
        self.d_model = d_model

        self.self_attn = MultiHeadAttention(num_heads=decoder_attention_heads)
        self.self_attn_layer_norm = LayerNormalization((d_model,))

        self.encoder_attn = MultiHeadAttention(num_heads=decoder_attention_heads)
        self.encoder_attn_layer_norm = LayerNormalization((d_model,))

        self.fc1 = Gemm(trans_b=1)
        self.act = Gelu()
        self.fc2 = Gemm(trans_b=1)
        self.final_layer_norm = LayerNormalization((d_model,))

    def __call__(  # noqa: D102
        self, x: Tensor, encoder_hidden_states: Tensor, causal_mask: Optional[Tensor] = None
    ) -> Tensor:
        identity = x
        x_norm = self.self_attn_layer_norm(
            x,
            get_param(f"{self.prefix}.self_attn_layer_norm.weight", [self.d_model]),
            get_param(f"{self.prefix}.self_attn_layer_norm.bias", [self.d_model]),
        )
        x_attn = self.self_attn(x_norm, x_norm, x_norm, mask=causal_mask)
        x = add(identity, x_attn)

        identity = x
        x_norm = self.encoder_attn_layer_norm(
            x,
            get_param(f"{self.prefix}.encoder_attn_layer_norm.weight", [self.d_model]),
            get_param(f"{self.prefix}.encoder_attn_layer_norm.bias", [self.d_model]),
        )
        x_attn = self.encoder_attn(x_norm, encoder_hidden_states, encoder_hidden_states)
        x = add(identity, x_attn)

        identity = x
        x_norm = self.final_layer_norm(
            x,
            get_param(f"{self.prefix}.final_layer_norm.weight", [self.d_model]),
            get_param(f"{self.prefix}.final_layer_norm.bias", [self.d_model]),
        )
        x_ffn = self.fc1(
            x_norm,
            get_param(f"{self.prefix}.fc1.weight", [self.d_model * 4, self.d_model]),
            get_param(f"{self.prefix}.fc1.bias", [self.d_model * 4]),
        )
        x_ffn = self.act(x_ffn)
        x_ffn = self.fc2(
            x_ffn,
            get_param(f"{self.prefix}.fc2.weight", [self.d_model, self.d_model * 4]),
            get_param(f"{self.prefix}.fc2.bias", [self.d_model]),
        )

        return add(identity, x_ffn)


class WhisperDecoder:  # noqa: D101
    def __init__(  # noqa: D107
        self,
        vocab_size: int = 51865,
        d_model: int = 512,
        decoder_attention_heads: int = 8,
        decoder_ffn_dim: int = 2048,
        decoder_layers: int = 6,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.layers = [
            WhisperDecoderLayer(
                d_model, decoder_attention_heads, decoder_ffn_dim, prefix=f"layers.{i}"
            )
            for i in range(decoder_layers)
        ]
        self.layer_norm = LayerNormalization((d_model,))
        self.lm_head = Gemm(trans_b=1)

    def __call__(  # noqa: D102
        self, input_ids: Tensor, encoder_hidden_states: Tensor, causal_mask: Optional[Tensor] = None
    ) -> Tensor:
        from onnx9000.core.ops import gather

        x = gather(
            get_param("embed_tokens.weight", [self.vocab_size, self.d_model]), input_ids, axis=0
        )

        pos_embed = get_param("embed_positions.weight", [1, 448, self.d_model])
        x = add(x, pos_embed)

        for layer in self.layers:
            x = layer(x, encoder_hidden_states, causal_mask)

        x = self.layer_norm(
            x,
            get_param("layer_norm.weight", [self.d_model]),
            get_param("layer_norm.bias", [self.d_model]),
        )
        x = self.lm_head(x, get_param("lm_head.weight", [self.vocab_size, self.d_model]))
        return x


class Whisper:  # noqa: D101
    def __init__(  # noqa: D107
        self,
        d_model: int = 512,
        encoder_attention_heads: int = 8,
        encoder_ffn_dim: int = 2048,
        encoder_layers: int = 6,
        decoder_attention_heads: int = 8,
        decoder_ffn_dim: int = 2048,
        decoder_layers: int = 6,
        vocab_size: int = 51865,
    ):
        self.encoder = WhisperEncoder(
            d_model=d_model,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_layers=encoder_layers,
        )
        self.decoder = WhisperDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_layers=decoder_layers,
        )

    def __call__(self, input_features: Tensor, decoder_input_ids: Tensor) -> Tensor:  # noqa: D102
        encoder_hidden_states = self.encoder(input_features)
        out = self.decoder(decoder_input_ids, encoder_hidden_states)
        return out


def whisper_tiny(**kwargs: Any) -> Whisper:  # noqa: D103
    return Whisper(
        d_model=384,
        encoder_attention_heads=6,
        encoder_ffn_dim=1536,
        encoder_layers=4,
        decoder_attention_heads=6,
        decoder_ffn_dim=1536,
        decoder_layers=4,
        **kwargs,
    )
