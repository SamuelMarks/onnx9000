"""Backward-compatible re-export module for `onnx9000_llama_web`."""

from onnx9000.llama_web import LlamaWebConfig, LlamaWebSession, run_model

__all__ = ["LlamaWebConfig", "LlamaWebSession", "run_model"]
