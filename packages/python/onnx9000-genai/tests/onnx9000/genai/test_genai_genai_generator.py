"""Unit tests for Generator in `onnx9000.genai.generator`."""

from onnx9000.core.ir import Graph
from onnx9000.genai.generator import Generator
from onnx9000.genai.state import ContinuousKVCache, State
from onnx9000.genai.types import GeneratorParams


def test_generator_instantiation() -> None:
    """Verify Generator initializes with State and GeneratorParams."""
    st = State(Graph("gen_g"), ContinuousKVCache())
    params = GeneratorParams(max_length=16)
    gen = Generator(st, params)
    assert gen.state is st
    assert gen.params is params
