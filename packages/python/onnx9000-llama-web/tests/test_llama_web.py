"""Comprehensive unit tests for onnx9000.llama_web and onnx9000_llama_web."""

import onnx9000_llama_web
import pytest
from onnx9000.llama_web import LlamaWebConfig, LlamaWebSession, run_model


def test_run_model_and_kv_session() -> None:
    """Verify run_model(), LlamaWebConfig.kv_cache_bytes(), and LlamaWebSession."""
    with pytest.raises(ValueError, match="Invalid model string"):
        run_model("")
    assert run_model("test") == "[LLaMA-Web] processing test"
    assert onnx9000_llama_web.run_model("llama3") == "[LLaMA-Web] processing llama3"

    cfg = LlamaWebConfig(
        model_id="llama-3-8b",
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        max_seq_len=8,
        dtype_bytes=2,
    )
    assert cfg.kv_cache_bytes() == 2 * 2 * 2 * 4 * 8 * 2
    assert cfg.kv_cache_bytes(seq_len=4) == 2 * 2 * 2 * 4 * 4 * 2

    session = LlamaWebSession(config=cfg)
    assert session.append_tokens([101, 102, 103]) == 3
    with pytest.raises(ValueError, match="Context length exceeded max_seq_len"):
        session.append_tokens([1, 2, 3, 4, 5, 6])
    session.clear()
    assert session.tokens == []
