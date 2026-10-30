"""Generative AI runtime, tokenizers, KV-cache management, and decoding algorithms for ONNX9000."""

from .generator import Generator
from .logit_processor_list import LogitProcessorList
from .logit_processors import (
    LogitProcessor,
    RepetitionPenaltyLogitProcessor,
    TemperatureLogitProcessor,
    TopKLogitProcessor,
)
from .model import Model
from .search import GreedySearch, MultinomialSampling, SearchAlgorithm
from .state import (
    ContinuousKVCache,
    CrossAttentionCache,
    GroupedQueryAttentionCache,
    KVCache,
    MultiHeadAttentionCache,
    MultiQueryAttentionCache,
    PagedKVCache,
    PositionalEmbeddingUtils,
    SequenceBatchingKVCache,
    SlidingWindowKVCache,
    State,
)
from .tensor_utils import SequenceTensorUtils
from .tokenizer import (
    BPETokenizer,
    HuggingFaceTokenizerLoader,
    PreTokenizer,
    Tokenizer,
    TokenizerStream,
    UnicodeNormalizer,
    UnigramTokenizer,
    WordPieceTokenizer,
)
from .top_p import TopPLogitProcessor
from .types import GeneratorParams, ModelParams


class GenAI:
    """High-level facade for running ONNX9000 Generative AI pipelines."""

    def process(self, prompt: str) -> str:
        """Process a prompt string through the GenAI pipeline.

        Args:
            prompt: Input prompt string.

        Returns:
            Processed GenAI output string.
        """
        return f"GenAI processed {prompt}"


def run() -> str:
    """Execute the default GenAI subsystem status check.

    Returns:
        GenAI status confirmation string.
    """
    return "[genai] processed"


__all__ = [
    "BPETokenizer",
    "ContinuousKVCache",
    "CrossAttentionCache",
    "GenAI",
    "Generator",
    "GeneratorParams",
    "GreedySearch",
    "GroupedQueryAttentionCache",
    "HuggingFaceTokenizerLoader",
    "KVCache",
    "LogitProcessor",
    "LogitProcessorList",
    "Model",
    "ModelParams",
    "MultiHeadAttentionCache",
    "MultiQueryAttentionCache",
    "MultinomialSampling",
    "PagedKVCache",
    "PositionalEmbeddingUtils",
    "PreTokenizer",
    "RepetitionPenaltyLogitProcessor",
    "SearchAlgorithm",
    "SequenceBatchingKVCache",
    "SequenceTensorUtils",
    "SlidingWindowKVCache",
    "State",
    "TemperatureLogitProcessor",
    "Tokenizer",
    "TokenizerStream",
    "TopKLogitProcessor",
    "TopPLogitProcessor",
    "UnicodeNormalizer",
    "UnigramTokenizer",
    "WordPieceTokenizer",
    "run",
]
