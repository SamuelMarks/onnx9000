"""Provide functionality for this module."""

import math

from ..core.ir import Tensor


class LogitProcessor:
    """Base class for modifying logits during generation."""

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        return logits


class TemperatureLogitProcessor(LogitProcessor):
    """Apply temperature scaling to logits."""

    def __init__(self, temperature: float):
        """Initialize the instance."""
        if temperature <= 0.0:
            raise ValueError("Temperature must be strictly positive.")
        self.temperature = temperature

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if self.temperature == 1.0 or logits.data is None:
            return logits

        import struct

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        new_data = bytearray(len(logits.data))

        # Simple inplace scaling for float32
        for i in range(0, len(logits.data), itemsize):
            val = struct.unpack("<f", logits.data[i : i + itemsize])[0]
            val = val / self.temperature
            new_data[i : i + itemsize] = struct.pack("<f", val)

        return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)


class TopKLogitProcessor(LogitProcessor):
    """Filters logits, retaining only the top K most likely tokens."""

    def __init__(self, top_k: int):
        """Initialize the instance."""
        if top_k <= 0:
            raise ValueError("top_k must be strictly positive.")
        self.top_k = top_k

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if logits.data is None:
            return logits

        import struct

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize

        offset = len(logits.data) - vocab_size * itemsize

        # Extract last token logits
        vals = []
        for i in range(vocab_size):
            start = offset + i * itemsize
            if start >= 0 and start + itemsize <= len(logits.data):
                val = struct.unpack("<f", logits.data[start : start + itemsize])[0]
                vals.append((val, i))

        # Sort and find threshold
        vals.sort(key=lambda x: x[0], reverse=True)
        if len(vals) > self.top_k:
            threshold = vals[self.top_k - 1][0]

            new_data = bytearray(logits.data)
            neg_inf = struct.pack("<f", float("-inf"))

            for i in range(vocab_size):
                start = offset + i * itemsize
                if start >= 0 and start + itemsize <= len(logits.data):
                    val = struct.unpack("<f", logits.data[start : start + itemsize])[0]
                    if val < threshold:
                        new_data[start : start + itemsize] = neg_inf

            return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)

        return logits


class RepetitionPenaltyLogitProcessor(LogitProcessor):
    """Apply a penalty to tokens that have already been generated."""

    def __init__(self, penalty: float):
        """Initialize the instance."""
        if penalty <= 0.0:
            raise ValueError("Penalty must be strictly positive.")
        self.penalty = penalty

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if self.penalty == 1.0 or logits.data is None or not input_ids:
            return logits

        import struct

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize

        offset = len(logits.data) - vocab_size * itemsize
        new_data = bytearray(logits.data)

        for token_id in set(input_ids):
            if token_id < vocab_size:
                start = offset + token_id * itemsize
                if start >= 0 and start + itemsize <= len(logits.data):
                    val = struct.unpack("<f", logits.data[start : start + itemsize])[0]
                    # if val < 0 penalty is applied as multiplication, else division
                    if val < 0:
                        val = val * self.penalty
                    else:
                        val = val / self.penalty
                    new_data[start : start + itemsize] = struct.pack("<f", val)

        return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)


class MinPLogitProcessor(LogitProcessor):
    """Filters logits based on min-p sampling."""

    def __init__(self, min_p: float):
        """Initialize the instance."""
        if min_p <= 0.0 or min_p > 1.0:
            raise ValueError("min_p must be strictly positive and <= 1.0")
        self.min_p = min_p

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if self.min_p >= 1.0 or logits.data is None:
            return logits

        import struct

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize

        offset = len(logits.data) - vocab_size * itemsize

        vals = []
        max_val = float("-inf")
        for i in range(vocab_size):
            start = offset + i * itemsize
            if start >= 0 and start + itemsize <= len(logits.data):
                val = struct.unpack("<f", logits.data[start : start + itemsize])[0]
                vals.append((val, i))
                if val > max_val:
                    max_val = val

        if max_val == float("-inf"):
            return logits

        threshold = max_val + math.log(self.min_p)

        new_data = bytearray(logits.data)
        neg_inf = struct.pack("<f", float("-inf"))

        for val, idx in vals:
            if val < threshold:
                start = offset + idx * itemsize
                new_data[start : start + itemsize] = neg_inf

        return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)


class PresencePenaltyLogitProcessor(LogitProcessor):
    """Apply a presence penalty to tokens that have already been generated."""

    def __init__(self, penalty: float):
        """Initialize the instance."""
        self.penalty = penalty

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if self.penalty == 0.0 or logits.data is None or not input_ids:
            return logits

        import struct

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize

        offset = len(logits.data) - vocab_size * itemsize
        new_data = bytearray(logits.data)

        for token_id in set(input_ids):
            if token_id < vocab_size:
                start = offset + token_id * itemsize
                if start >= 0 and start + itemsize <= len(logits.data):
                    val = struct.unpack("<f", logits.data[start : start + itemsize])[0]
                    val = val - self.penalty
                    new_data[start : start + itemsize] = struct.pack("<f", val)

        return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)


class FrequencyPenaltyLogitProcessor(LogitProcessor):
    """Apply a frequency penalty to tokens that have already been generated."""

    def __init__(self, penalty: float):
        """Initialize the instance."""
        self.penalty = penalty

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if self.penalty == 0.0 or logits.data is None or not input_ids:
            return logits

        import struct
        from collections import Counter

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize

        counts = Counter(input_ids)
        offset = len(logits.data) - vocab_size * itemsize
        new_data = bytearray(logits.data)

        for token_id, count in counts.items():
            if token_id < vocab_size:
                start = offset + token_id * itemsize
                if start >= 0 and start + itemsize <= len(logits.data):
                    val = struct.unpack("<f", logits.data[start : start + itemsize])[0]
                    val = val - self.penalty * count
                    new_data[start : start + itemsize] = struct.pack("<f", val)

        return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)


class ForcedBOSLogitProcessor(LogitProcessor):
    """Forces the first generated token to be the BOS token."""

    def __init__(self, bos_token_id: int):
        """Initialize the instance."""
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if len(input_ids) == 0 and logits.data is not None:
            import struct

            itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
            vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize
            offset = len(logits.data) - vocab_size * itemsize
            new_data = bytearray(logits.data)
            neg_inf = struct.pack("<f", float("-inf"))
            for i in range(vocab_size):
                if i != self.bos_token_id:
                    start = offset + i * itemsize
                    if start >= 0 and start + itemsize <= len(logits.data):
                        new_data[start : start + itemsize] = neg_inf
            return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)
        return logits


class ForcedEOSLogitProcessor(LogitProcessor):
    """Forces the token to be EOS when max_length is reached."""

    def __init__(self, max_length: int, eos_token_id: int):
        """Initialize the instance."""
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if len(input_ids) == self.max_length - 1 and logits.data is not None:
            import struct

            itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
            vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize
            offset = len(logits.data) - vocab_size * itemsize
            new_data = bytearray(logits.data)
            neg_inf = struct.pack("<f", float("-inf"))
            for i in range(vocab_size):
                if i != self.eos_token_id:
                    start = offset + i * itemsize
                    if start >= 0 and start + itemsize <= len(logits.data):
                        new_data[start : start + itemsize] = neg_inf
            return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)
        return logits


class LogitBiasProcessor(LogitProcessor):
    """Apply custom biases to specific tokens."""

    def __init__(self, bias_map: dict[int, float]):
        """Initialize the instance."""
        self.bias_map = bias_map

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if not self.bias_map or logits.data is None:
            return logits

        import struct

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize

        offset = len(logits.data) - vocab_size * itemsize
        new_data = bytearray(logits.data)

        for token_id, bias in self.bias_map.items():
            if token_id < vocab_size:
                start = offset + token_id * itemsize
                if start >= 0 and start + itemsize <= len(logits.data):
                    val = struct.unpack("<f", logits.data[start : start + itemsize])[0]
                    new_data[start : start + itemsize] = struct.pack("<f", val + bias)

        return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)


class NoRepeatNGramLogitProcessor(LogitProcessor):
    """Prevents generating n-grams that have already been generated."""

    def __init__(self, ngram_size: int):
        """Initialize the instance."""
        if ngram_size <= 0:
            raise ValueError("ngram_size must be strictly positive")
        self.ngram_size = ngram_size

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if self.ngram_size == 0 or len(input_ids) < self.ngram_size - 1 or logits.data is None:
            return logits

        import struct

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize

        offset = len(logits.data) - vocab_size * itemsize

        prefix = input_ids[-(self.ngram_size - 1) :] if self.ngram_size > 1 else []
        banned_tokens = set()

        for i in range(len(input_ids) - self.ngram_size + 1):
            if self.ngram_size == 1 or input_ids[i : i + self.ngram_size - 1] == prefix:
                banned_tokens.add(input_ids[i + self.ngram_size - 1])

        if not banned_tokens:
            return logits

        new_data = bytearray(logits.data)
        neg_inf = struct.pack("<f", float("-inf"))

        for token_id in banned_tokens:
            if token_id < vocab_size:
                start = offset + token_id * itemsize
                if start >= 0 and start + itemsize <= len(logits.data):
                    new_data[start : start + itemsize] = neg_inf

        return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)


class NoBadWordsLogitProcessor(LogitProcessor):
    """Prevents generating specific token sequences."""

    def __init__(self, bad_words_ids: list[list[int]]):
        """Initialize the instance."""
        self.bad_words_ids = bad_words_ids

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if not self.bad_words_ids or logits.data is None:
            return logits

        import struct

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize

        offset = len(logits.data) - vocab_size * itemsize
        banned_tokens = set()

        for bad_word in self.bad_words_ids:
            if len(bad_word) == 1:
                banned_tokens.add(bad_word[0])
            elif len(bad_word) > 1:
                prefix = bad_word[:-1]
                if len(input_ids) >= len(prefix) and input_ids[-len(prefix) :] == prefix:
                    banned_tokens.add(bad_word[-1])

        if not banned_tokens:
            return logits

        new_data = bytearray(logits.data)
        neg_inf = struct.pack("<f", float("-inf"))

        for token_id in banned_tokens:
            if token_id < vocab_size:
                start = offset + token_id * itemsize
                if start >= 0 and start + itemsize <= len(logits.data):
                    new_data[start : start + itemsize] = neg_inf

        return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)


class AllowedWordsLogitProcessor(LogitProcessor):
    """Restricts the generated tokens to a specific set."""

    def __init__(self, allowed_token_ids: list[int]):
        """Initialize the instance."""
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids: list[int], logits: Tensor) -> Tensor:
        """Execute the callable."""
        if not self.allowed_token_ids or logits.data is None:
            return logits

        import struct

        itemsize = logits.dtype.itemsize if hasattr(logits.dtype, "itemsize") else 4
        vocab_size = logits.shape[-1] if logits.shape else len(logits.data) // itemsize

        offset = len(logits.data) - vocab_size * itemsize
        new_data = bytearray(logits.data)
        neg_inf = struct.pack("<f", float("-inf"))

        for i in range(vocab_size):
            if i not in self.allowed_token_ids:
                start = offset + i * itemsize
                if start >= 0 and start + itemsize <= len(logits.data):
                    new_data[start : start + itemsize] = neg_inf

        return Tensor(name=logits.name, shape=logits.shape, dtype=logits.dtype, data=new_data)
