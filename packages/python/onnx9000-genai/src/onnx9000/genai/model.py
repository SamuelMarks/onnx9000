"""Implementation of model (onnx9000 / genai / model.py) providing Model."""

from collections.abc import AsyncIterator

from onnx9000.core.ir import Tensor

from .generator import Generator
from .tokenizer import Tokenizer
from .types import GeneratorParams, ModelParams

_RUNTIME_TYPES = (AsyncIterator, Tensor, Generator, Tokenizer, GeneratorParams, ModelParams)


class Model:
    """Base Model class for GenAI wrappers."""

    def __init__(self, params: ModelParams) -> None:
        """Initialize Model with configured parameters and state.

        Args:
            params: Parameter specifying params (ModelParams).
        """
        self.params = params

    def create_tokenizer(self) -> Tokenizer:
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
