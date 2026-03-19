"""Provide functionality for this module."""

from collections.abc import AsyncIterator

from ..core.ir import Tensor
from .generator import Generator
from .state import State
from .types import GeneratorParams, ModelParams


class Model:
    """Base Model class for GenAI wrappers."""

    def __init__(self, params: ModelParams):
        """Initialize the instance."""
        self.params = params

    def create_tokenizer(self) -> "Tokenizer":
        """Execute the create_tokenizer operation."""
        from .tokenizer import Tokenizer

        return Tokenizer()

    def create_generator(self, params: GeneratorParams) -> Generator:
        """Execute the create_generator operation."""
        from .generator import Generator

        return Generator(None, params)

    async def generate(self, prompt_ids: Tensor, params: GeneratorParams) -> AsyncIterator[int]:
        """High-level generate API."""
        generator = self.create_generator(params)
        async for token in generator.generate(prompt_ids):
            yield token
