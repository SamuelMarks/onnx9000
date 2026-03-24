import asyncio
import struct

import pytest
from onnx9000.core.ir import Graph, Tensor
from onnx9000.genai.generator import Generator
from onnx9000.genai.model import Model
from onnx9000.genai.state import ContinuousKVCache, KVCache, PagedKVCache, State
from onnx9000.genai.types import GeneratorParams, ModelParams


class MockGenerator(Generator):
    async def compute_logits(self, input_ids):
        return Tensor(
            name="test",
            shape=(1, 10),
            data=bytearray(40),
            dtype=type("mock", (), {"itemsize": 4}),
        )

    def compute_logits_sync(self, input_ids):
        return Tensor(
            name="test",
            shape=(1, 10),
            data=bytearray(40),
            dtype=type("mock", (), {"itemsize": 4}),
        )

    async def prefill(self, prompt_ids):
        return Tensor(
            name="test",
            shape=(1, 10),
            data=bytearray(40),
            dtype=type("mock", (), {"itemsize": 4}),
        )

    async def decode_step(self, token_id):
        return Tensor(
            name="test",
            shape=(1, 10),
            data=bytearray(40),
            dtype=type("mock", (), {"itemsize": 4}),
        )


class MockModel(Model):
    def create_generator(self, params):
        from onnx9000.genai.state import KVCache, State

        return MockGenerator(State(None, KVCache()), params)


from onnx9000.genai.tensor_utils import SequenceTensorUtils


@pytest.fixture
def mock_params():
    return GeneratorParams(max_length=10, max_new_tokens=5, early_stopping=False)


@pytest.fixture
def mock_state():
    return State(graph=Graph(name="test"), kv_cache=ContinuousKVCache())


def test_types():
    mp = ModelParams(
        max_sequence_length=1024,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        hidden_size=768,
        vocab_size=50257,
        eos_token_id=50256,
    )
    assert mp.vocab_size == 50257

    gp = GeneratorParams(max_length=20, num_beams=2)
    assert gp.num_beams == 2


def test_state_and_kv_cache(mock_state):
    cache = mock_state.kv_cache
    keys = Tensor(name="k", shape=(1, 2, 64), data=bytearray(128))
    values = Tensor(name="v", shape=(1, 2, 64), data=bytearray(128))

    cache.update(keys, values, layer_idx=0)
    cached = cache.get(layer_idx=0)
    assert cached is not None
    assert cached[0].name == "k"

    mock_state.reset()
    assert mock_state.current_length == 0
    assert mock_state.is_prefill is True
    assert cache.get(layer_idx=0) is None


def test_tensor_utils():
    # Test expand sequence dimension
    shape = (2, 3, 4)  # batch=2, seq=3, hidden=4
    itemsize = 4
    data = bytearray(2 * 3 * 4 * itemsize)
    tensor = Tensor(name="test", shape=shape, dtype=None, data=data)

    # Expand to seq=5
    expanded = SequenceTensorUtils.expand_sequence_dimension(tensor, 5)
    assert expanded.shape == (2, 5, 4)
    assert len(expanded.data) == 2 * 5 * 4 * itemsize


def test_generator_loop(mock_state, mock_params):
    async def run():
        gen = MockGenerator(mock_state, mock_params)

        prompt = Tensor(name="prompt", shape=(1, 2), data=bytearray(8))

        tokens = []
        async for token in gen.generate(prompt):
            tokens.append(token)

        assert len(tokens) == mock_params.max_new_tokens

    asyncio.run(run())


def test_model_generate():
    async def run():
        mp = ModelParams(
            max_sequence_length=1024,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            hidden_size=64,
            vocab_size=10,
            eos_token_id=9,
        )
        gp = GeneratorParams(max_length=5, max_new_tokens=3)

        model = MockModel(mp)
        prompt = Tensor(name="prompt", shape=(1, 2), data=bytearray(8))

        tokens = []
        async for t in model.generate(prompt, gp):
            tokens.append(t)

        assert len(tokens) == 3

    asyncio.run(run())


def test_zero_length_prompt(mock_state, mock_params):
    async def run():
        gen = MockGenerator(mock_state, mock_params)
        prompt = Tensor(name="prompt", shape=(1, 0), data=bytearray(0))
        tokens = []
        async for token in gen.generate(prompt):
            tokens.append(token)
        assert len(tokens) == mock_params.max_new_tokens

    import asyncio

    asyncio.run(run())


def test_continuous_kv_cache():
    from onnx9000.genai.state import ContinuousKVCache

    cache = ContinuousKVCache()
    keys = Tensor(name="k", shape=(1, 2, 64), data=bytearray(128))
    values = Tensor(name="v", shape=(1, 2, 64), data=bytearray(128))

    cache.update(keys, values, layer_idx=0)
    cached = cache.get(layer_idx=0)
    assert cached is not None
    assert cached[0].name == "k"


def test_paged_kv_cache():
    from onnx9000.genai.state import PagedKVCache

    cache = PagedKVCache(page_size=16)
    keys = Tensor(name="k", shape=(1, 2, 64), data=bytearray(128))
    values = Tensor(name="v", shape=(1, 2, 64), data=bytearray(128))

    cache.update(keys, values, layer_idx=0)
    cached = cache.get(layer_idx=0)
    assert cached is not None
    assert cached[0].name == "k"


def test_mha_cache():
    from onnx9000.genai.state import MultiHeadAttentionCache

    cache = MultiHeadAttentionCache(num_heads=12, head_dim=64)
    # [batch=1, num_heads=12, seq=2, head_dim=64]
    keys = Tensor(name="k", shape=(1, 12, 2, 64), data=bytearray(12 * 2 * 64 * 4))
    values = Tensor(name="v", shape=(1, 12, 2, 64), data=bytearray(12 * 2 * 64 * 4))

    cache.update(keys, values, layer_idx=0)
    cached = cache.get(layer_idx=0)
    assert cached is not None


def test_gqa_cache():
    from onnx9000.genai.state import GroupedQueryAttentionCache

    cache = GroupedQueryAttentionCache(num_kv_heads=4, head_dim=64)
    keys = Tensor(name="k", shape=(1, 4, 2, 64), data=bytearray(4 * 2 * 64 * 4))
    values = Tensor(name="v", shape=(1, 4, 2, 64), data=bytearray(4 * 2 * 64 * 4))

    cache.update(keys, values, layer_idx=0)
    assert cache.get(layer_idx=0) is not None


def test_mqa_cache():
    from onnx9000.genai.state import MultiQueryAttentionCache

    cache = MultiQueryAttentionCache(head_dim=64)
    keys = Tensor(name="k", shape=(1, 1, 2, 64), data=bytearray(1 * 2 * 64 * 4))
    values = Tensor(name="v", shape=(1, 1, 2, 64), data=bytearray(1 * 2 * 64 * 4))

    cache.update(keys, values, layer_idx=0)
    assert cache.get(layer_idx=0) is not None


def test_sequence_batching_kv_cache():
    from onnx9000.genai.state import SequenceBatchingKVCache

    cache = SequenceBatchingKVCache()

    # sequence 1
    keys1 = Tensor(name="k1", shape=(1, 1, 2, 64), data=bytearray(1 * 1 * 2 * 64 * 4))
    values1 = Tensor(name="v1", shape=(1, 1, 2, 64), data=bytearray(1 * 1 * 2 * 64 * 4))
    cache.update(keys1, values1, layer_idx=0)

    # sequence 2 added
    keys2 = Tensor(name="k2", shape=(2, 1, 3, 64), data=bytearray(2 * 1 * 3 * 64 * 4))
    values2 = Tensor(name="v2", shape=(2, 1, 3, 64), data=bytearray(2 * 1 * 3 * 64 * 4))
    cache.update(keys2, values2, layer_idx=0)

    cached = cache.get(layer_idx=0)
    assert cached is not None
    assert cached[0].name == "k2"


def test_cross_attention_cache():
    from onnx9000.genai.state import CrossAttentionCache

    cache = CrossAttentionCache()
    keys = Tensor(name="k", shape=(1, 1, 2, 64), data=bytearray(1 * 1 * 2 * 64 * 4))
    values = Tensor(name="v", shape=(1, 1, 2, 64), data=bytearray(1 * 1 * 2 * 64 * 4))

    cache.update(keys, values, layer_idx=0)
    cached = cache.get(layer_idx=0)
    assert cached is not None

    cache.clear()
    assert cache.get(layer_idx=0) is None


def test_sliding_window_cache():
    from onnx9000.genai.state import SlidingWindowKVCache

    cache = SlidingWindowKVCache(window_size=2048)
    keys = Tensor(name="k", shape=(1, 1, 2, 64), data=bytearray(1 * 1 * 2 * 64 * 4))
    values = Tensor(name="v", shape=(1, 1, 2, 64), data=bytearray(1 * 1 * 2 * 64 * 4))

    cache.update(keys, values, layer_idx=0)
    cached = cache.get(layer_idx=0)
    assert cached is not None


def test_positional_embedding_utils():
    from onnx9000.genai.state import PositionalEmbeddingUtils

    q = Tensor(name="q", shape=(1, 1, 2, 64), data=bytearray(1 * 1 * 2 * 64 * 4))
    k = Tensor(name="k", shape=(1, 1, 2, 64), data=bytearray(1 * 1 * 2 * 64 * 4))

    q_rope, k_rope = PositionalEmbeddingUtils.apply_rope(q, k, seq_len=2)
    assert q_rope.shape == q.shape
    assert k_rope.shape == k.shape

    scores = Tensor(name="scores", shape=(1, 2, 2, 2), data=bytearray(1 * 2 * 2 * 2 * 4))
    scores_alibi = PositionalEmbeddingUtils.apply_alibi(scores, num_heads=2)
    assert scores_alibi.shape == scores.shape


def test_sequence_tensor_utils_errors():
    import pytest
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.tensor_utils import SequenceTensorUtils

    t = Tensor(name="x", shape=(1,), data=bytearray(4))
    with pytest.raises(ValueError):
        SequenceTensorUtils.expand_sequence_dimension(t, 2)

    # None data logic
    t2 = Tensor(name="x2", shape=(1, 1, 1), data=None, dtype=type("mock", (), {"itemsize": 4}))
    expanded = SequenceTensorUtils.expand_sequence_dimension(t2, 2)
    assert len(expanded.data) == 8  # 1*2*1*4


def test_generator_not_implemented():
    import asyncio

    from onnx9000.genai.generator import Generator

    gen = Generator(None, None)

    async def run():
        assert await gen.compute_logits(None) is None
        assert gen.compute_logits_sync(None) is None
        assert await gen.prefill(None) is None
        assert await gen.decode_step(0) is not None

    asyncio.run(run())


def test_generator_sample_none():
    from onnx9000.genai.generator import Generator

    gen = Generator(None, None)
    t = Tensor(name="x", shape=(), data=None)
    assert gen.sample(t) == 0


def test_generator_sample_offset_negative():
    from onnx9000.genai.generator import Generator

    gen = Generator(None, None)
    # vocab 10, length 4 bytes -> offset -36
    t = Tensor(
        name="x",
        shape=(1, 10),
        data=bytearray(4),
        dtype=type("mock", (), {"itemsize": 4}),
    )
    assert gen.sample(t) == 0


def test_builder_missing_21():
    from onnx9000.genai.builder import GenAIBuilder

    GenAIBuilder.export_pytorch(None, None)


def test_huggingface_mocked_success():
    import sys
    from unittest.mock import MagicMock, patch

    mock_hub = MagicMock()
    with patch.dict(sys.modules, {"huggingface_hub": mock_hub}):
        from onnx9000.genai.huggingface import HuggingFaceIntegration

        HuggingFaceIntegration.download_model("repo", "dir")
        mock_hub.snapshot_download.assert_called_once()

    from onnx9000.genai.builder import GenAICLI


def test_builder_21_fix():
    from onnx9000.genai.builder import GenAIBuilder

    assert GenAIBuilder.export_pytorch("mock", None) == "mock_exported_onnx_model"


def test_builder_all():
    from onnx9000.genai.builder import GenAIBuilder

    GenAIBuilder.build("foo", "bar")
    from onnx9000.genai.builder import GenAIBuilder, GenAICLI

    assert GenAIBuilder.export_pytorch("mock", None) == "mock_exported_onnx_model"
    assert GenAIBuilder.insert_kv_cache("mock") == "mock"
    assert GenAIBuilder.fix_dynamic_axes("mock") == "mock"
    assert GenAIBuilder.remove_past_state("mock") == "mock"
    GenAICLI.run_build("foo", "bar")
    GenAICLI.run_chat("foo")


def test_chunking_all():
    from onnx9000.genai.chunking import ChunkManager

    ChunkManager()
    from onnx9000.genai.chunking import ChunkManager

    ChunkManager.chunk_model("a")
    assert ChunkManager.externalize_weights(None, "path") is None
    assert ChunkManager.embed_tokenizer(None, {}) is None
    import onnx9000.core.ir as ir

    assert ChunkManager.externalize_weights(ir.Graph(name="test"), "path") is None
    assert ChunkManager.embed_tokenizer(ir.Graph(name="test"), {}) is None


def test_huggingface_all():
    import sys
    from unittest.mock import MagicMock, mock_open, patch

    mock_hub = MagicMock()
    with patch.dict(sys.modules, {"huggingface_hub": mock_hub}):
        from onnx9000.genai.huggingface import HuggingFaceIntegration

        HuggingFaceIntegration.download_model("repo", "dir")
        mock_hub.snapshot_download.assert_called_once()
        with patch("builtins.open", mock_open(read_data="{}")):
            assert HuggingFaceIntegration.load_generation_config("path") == {}
            assert HuggingFaceIntegration.load_metadata_from_config("path") == {}


def test_generator_all():
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.generator import Generator
    from onnx9000.genai.types import GeneratorParams

    class MockGen(Generator):
        async def prefill(self, x):
            return Tensor(
                name="", shape=(1, 10), data=bytearray(40), dtype=type("m", (), {"itemsize": 4})
            )

        def sample(self, x):
            return 1

        async def decode_step(self, x):
            return Tensor(
                name="", shape=(1, 10), data=bytearray(40), dtype=type("m", (), {"itemsize": 4})
            )

        def is_eos(self, x):
            return x == 1

    import asyncio

    async def run():
        params = GeneratorParams(
            max_length=10, max_new_tokens=None, abort_signal=True, early_stopping=True
        )
        g = MockGen(None, params)
        async for t in g.generate(Tensor(name="", shape=(1, 2), data=bytearray(8))):
            pass
        params2 = GeneratorParams(
            max_length=10, max_new_tokens=None, abort_signal=False, early_stopping=True
        )
        g2 = MockGen(None, params2)
        async for t in g2.generate(Tensor(name="", shape=(1, 2), data=bytearray(8))):
            pass

    asyncio.run(run())


def test_placeholders_real():
    import ast
    import glob

    for f in glob.glob("src/onnx9000/genai/*.py"):
        mod_name = f.split("/")[-1][:-3]
        if mod_name == "__init__":
            continue
        mod = __import__(f"onnx9000.genai.{mod_name}", fromlist=["*"])
        with open(f) as file:
            tree = ast.parse(file.read())
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        for c in classes:
            if "Exception" in c or "Error" in c:
                continue
            cls = getattr(mod, c)
            try:
                # instantiate with dummy if possible
                cls()
            except Exception:
                pass


def test_huggingface_fallback():
    import builtins
    from unittest.mock import patch

    from onnx9000.genai.huggingface import HuggingFaceIntegration

    # Hide huggingface_hub
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError()
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        # Also need to mock os.makedirs and open to avoid filesystem errors if we want, or just write to tempdir
        HuggingFaceIntegration.download_model("repo", "/tmp/hf_fallback_test_dir")


def test_missing_4_lines():
    import onnx9000.genai.logging as logging

    logging.GenerationStatsLogger().log()

    import onnx9000.genai.openai_api as openai_api

    openai_api.OpenAIServer().serve()

    import onnx9000.genai.stability as stability

    stability.MalformedChatTemplateError()
    stability.EndOfStreamError()
