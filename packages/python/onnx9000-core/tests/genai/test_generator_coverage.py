"""Test coverage gaps for the Generator class in the GenAI module."""

import asyncio

import pytest
from onnx9000.core.ir import Tensor
from onnx9000.genai.generator import Generator
from onnx9000.genai.types import GeneratorParams


class MockGenerator(Generator):
    """A mock generator for testing edge cases in the generate loop."""

    def __init__(self, state, params, eos_id=None):
        """Initialize with optional EOS ID."""
        super().__init__(state, params)
        self.eos_id = eos_id

    async def prefill(self, prompt_ids):
        """Mock prefill returning dummy logits."""
        return Tensor(name="logits", shape=(1, 10), data=bytearray(40))

    async def decode_step(self, token_id):
        """Mock decode step returning dummy logits."""
        return Tensor(name="logits", shape=(1, 10), data=bytearray(40))

    def sample(self, logits):
        """Mock sample returning a fixed token ID."""
        return 1

    def is_eos(self, token_id):
        """Check if token matches the designated EOS ID."""
        return token_id == self.eos_id


def test_generator_abort_signal():
    """Verify that the generation loop honors the abort_signal in GeneratorParams."""

    async def run():
        """Run."""
        params = GeneratorParams(max_length=100, max_new_tokens=10, abort_signal=True)
        gen = MockGenerator(None, params)
        prompt = Tensor(name="prompt", shape=(1, 1), data=bytearray(4))

        tokens = []
        async for token in gen.generate(prompt):
            tokens.append(token)

        # Should only yield the first token from prefill, then abort
        assert len(tokens) == 1

    asyncio.run(run())


def test_generator_early_stopping():
    """Verify that generation stops early when an EOS token is encountered."""

    async def run():
        """Run."""
        params = GeneratorParams(max_length=100, max_new_tokens=10, early_stopping=True)
        # Mock generator will return 1, and we set 1 as EOS
        gen = MockGenerator(None, params, eos_id=1)
        prompt = Tensor(name="prompt", shape=(1, 1), data=bytearray(4))

        tokens = []
        async for token in gen.generate(prompt):
            tokens.append(token)

        # Should yield the first token (which is 1/EOS), then stop
        assert len(tokens) == 1
        assert tokens[0] == 1

    asyncio.run(run())


def test_generator_max_tokens_none():
    """Verify max_tokens calculation when max_new_tokens is not provided."""

    async def run():
        # max_length=5, prompt_len=2 -> max_new_tokens should be 3
        """Run."""
        params = GeneratorParams(max_length=5, max_new_tokens=None)
        gen = MockGenerator(None, params)
        prompt = Tensor(name="prompt", shape=(1, 2), data=bytearray(8))

        tokens = []
        async for token in gen.generate(prompt):
            tokens.append(token)

        assert len(tokens) == 3

    asyncio.run(run())
