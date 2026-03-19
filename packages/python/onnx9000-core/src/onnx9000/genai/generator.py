import struct
from collections.abc import AsyncIterator, Iterator

from ..core.ir import Tensor
from .state import State
from .types import GeneratorParams


class Generator:
    """Base Generator class for stateful decoding."""

    def __init__(self, state: State, params: GeneratorParams):
        self.state = state
        self.params = params

    async def compute_logits(self, input_ids: Tensor) -> Tensor:
        """Compute logits for the current state."""
        raise NotImplementedError()

    def compute_logits_sync(self, input_ids: Tensor) -> Tensor:
        """Compute logits synchronously (if supported)."""
        raise NotImplementedError()

    async def prefill(self, prompt_ids: Tensor) -> Tensor:
        """Process pre-fill phase."""
        raise NotImplementedError()

    async def decode_step(self, token_id: int) -> Tensor:
        """Process a single decoding step."""
        raise NotImplementedError()

    async def generate(self, prompt_ids: Tensor) -> AsyncIterator[int]:
        """High-level generation API. Yields tokens as they are generated."""
        current_tokens = 0
        prompt_len = prompt_ids.shape[-1] if prompt_ids.shape else 0

        max_tokens = self.params.max_new_tokens
        if max_tokens is None:
            max_tokens = self.params.max_length - prompt_len

        logits = await self.prefill(prompt_ids)
        next_token = self.sample(logits)

        yield next_token
        current_tokens += 1

        while current_tokens < max_tokens:
            if self.params.abort_signal:
                break
            if self.params.early_stopping and self.is_eos(next_token):
                break

            logits = await self.decode_step(next_token)
            next_token = self.sample(logits)

            yield next_token
            current_tokens += 1

    def sample(self, logits: Tensor) -> int:
        """Sample the next token from logits."""
        # Simple argmax greedy search
        if logits.data is None:
            return 0

        vocab_size = logits.shape[-1] if logits.shape else 1
        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4

        # Assuming float32 for simplicity
        offset = len(logits.data) - vocab_size * itemsize
        if offset < 0:
            return 0

        max_val = float("-inf")
        max_idx = -1

        for i in range(vocab_size):
            start = offset + i * itemsize
            if start + itemsize <= len(logits.data):
                val = struct.unpack("<f", logits.data[start : start + itemsize])[0]
                if val > max_val:
                    max_val = val
                    max_idx = i

        return max_idx if max_idx != -1 else 0

    def is_eos(self, token_id: int) -> bool:
        return False
