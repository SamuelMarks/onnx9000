"""LLaMA Web/WASM configuration and KV-cache session packaging for ONNX9000."""

from dataclasses import dataclass, field

from onnx9000.core.exceptions import BackendExecutionError


@dataclass(frozen=True)
class LlamaWebConfig:
    """Configuration for browser-based WebGPU/WASM LLaMA inference.

    Attributes:
        model_id: Identifier or URL of the LLaMA ONNX model.
        num_layers: Number of transformer decoder layers.
        num_kv_heads: Number of key/value attention heads.
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum context window length in tokens.
        dtype_bytes: Bytes per KV-cache element (2 for FP16, 1 for INT8).
    """

    model_id: str
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    max_seq_len: int = 4096
    dtype_bytes: int = 2

    def kv_cache_bytes(self, seq_len: int | None = None) -> int:
        """Compute total KV-cache memory footprint in bytes.

        Args:
            seq_len: Sequence length to calculate for; defaults to `max_seq_len`.

        Returns:
            Total byte size of key and value caches across all layers.
        """
        effective_seq = self.max_seq_len if seq_len is None else seq_len
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * effective_seq
            * self.dtype_bytes
        )


@dataclass
class LlamaWebSession:
    """Stateful browser/WASM session tracker for LLaMA token generation."""

    config: LlamaWebConfig
    tokens: list[int] = field(default_factory=list)

    def append_tokens(self, new_tokens: list[int]) -> int:
        """Append token IDs to the active KV-cache session context.

        Args:
            new_tokens: Token IDs to append.

        Returns:
            Updated total token count in the session.

        Raises:
            ValueError: If appending exceeds `config.max_seq_len`.
        """
        if len(self.tokens) + len(new_tokens) > self.config.max_seq_len:
            raise BackendExecutionError("Context length exceeded max_seq_len")
        self.tokens.extend(new_tokens)
        return len(self.tokens)

    def clear(self) -> None:
        """Reset the session token buffer."""
        self.tokens.clear()


def run_model(model_str: str) -> str:
    """Run a LLaMA-Web model session descriptor.

    Args:
        model_str: Non-empty model identifier string.

    Returns:
        Formatted LLaMA-Web processing summary string.

    Raises:
        ValueError: If `model_str` is empty.
    """
    if not model_str:
        raise BackendExecutionError("Invalid model string")
    return f"[LLaMA-Web] processing {model_str}"


__all__ = ["LlamaWebConfig", "LlamaWebSession", "run_model"]
