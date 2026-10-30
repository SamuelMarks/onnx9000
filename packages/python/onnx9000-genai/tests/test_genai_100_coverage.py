"""Exhaustive 100% coverage tests for onnx9000-genai and onnx9000-sphinx-demo."""

import asyncio
import struct
import types
from pathlib import Path

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Tensor
from onnx9000.genai.builder import GenAIBuilder, GenAICLI
from onnx9000.genai.chunking import ChunkManager
from onnx9000.genai.ecosystem import V1Certification
from onnx9000.genai.extended import MultiTurnCache
from onnx9000.genai.generator import Generator
from onnx9000.genai.huggingface import HuggingFaceIntegration
from onnx9000.genai.logit_processors import (
    AllowedWordsLogitProcessor,
    ForcedBOSLogitProcessor,
    ForcedEOSLogitProcessor,
    FrequencyPenaltyLogitProcessor,
    LogitBiasProcessor,
    MinPLogitProcessor,
    NoBadWordsLogitProcessor,
    NoRepeatNGramLogitProcessor,
    PresencePenaltyLogitProcessor,
    RepetitionPenaltyLogitProcessor,
    TemperatureLogitProcessor,
    TopKLogitProcessor,
)
from onnx9000.genai.model import Model
from onnx9000.genai.qa import SamplingConfigLinter
from onnx9000.genai.search import SearchAlgorithm
from onnx9000.genai.state import (
    PagedKVCache,
    PositionalEmbeddingUtils,
    SequenceBatchingKVCache,
    State,
)
from onnx9000.genai.tensor_utils import SequenceTensorUtils
from onnx9000.genai.tokenizer import (
    BPETokenizer,
    Tokenizer,
    UnigramTokenizer,
    WordPieceTokenizer,
)
from onnx9000.genai.types import GeneratorParams, ModelParams
from onnx9000.sphinx_demo import copy_frontend_assets, setup


def test_genai_remaining_lines_and_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover all remaining lines and branches in onnx9000-genai."""

    async def _run() -> None:
        """Test helper function _run."""
        import sys

        # builder.py, chunking.py, ecosystem.py, extended.py, huggingface.py, qa.py, tensor_utils.py
        assert GenAIBuilder.export_pytorch("model", "input") == "mock_exported_onnx_model"
        GenAICLI.run_build("m1", "webgpu")
        ChunkManager.externalize_weights(None, "out")
        ChunkManager.embed_tokenizer(None, {})
        cert = V1Certification()
        cert.certify("m1")
        cert.certify("m1")  # already in certified_models
        mtc = MultiTurnCache()
        mtc.update("s1", 1)
        mtc.update("s1", 2)  # session_id already in cache
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        HuggingFaceIntegration.download_model("org/m", str(tmp_path / "hf"))
        monkeypatch.undo()
        assert SamplingConfigLinter().lint({"temperature": 0.5}) == []
        t_zero_batch = Tensor("zb", (0, 2), DType.FLOAT32, data=None)
        assert SequenceTensorUtils.expand_sequence_dimension(t_zero_batch, 4).shape == (0, 4)

        # model.py & generator.py
        mp = ModelParams(10, 2, 2, 2, 8, 100, 2)
        gp = GeneratorParams(max_length=10)
        m = Model(mp)
        assert m.params is mp
        assert isinstance(m.create_generator(gp), Generator)
        assert isinstance(m.create_tokenizer(), Tokenizer)

        from collections.abc import AsyncIterator

        async def fake_gen(_prompt: Tensor) -> AsyncIterator[int]:
            """Test helper function fake_gen."""
            yield 1
            yield 2

        monkeypatch.setattr(
            m, "create_generator", lambda _p: types.SimpleNamespace(generate=fake_gen)
        )
        tokens = [tok async for tok in m.generate(Tensor("p", (1, 1), DType.INT64), gp)]
        assert tokens == [1, 2]
        monkeypatch.undo()

        # generator.py: State base methods, TemperatureLogitProcessor, Generator
        st_base = State(Graph("g"), PagedKVCache())
        t_short_logits = Tensor("sl", (1, 10), DType.FLOAT32, data=struct.pack("<f", 1.0))
        TemperatureLogitProcessor(1.0)([1], t_short_logits)

        gen = Generator(st_base, gp)
        assert await gen.compute_logits(t_short_logits) is None
        assert gen.compute_logits_sync(t_short_logits) is None
        assert await gen.prefill(t_short_logits) is None
        assert (await gen.decode_step(1)).name == "dummy"
        assert gen.sample(Tensor("l_none", (1, 2), DType.FLOAT32, data=None)) == 0
        assert gen.sample(Tensor("l_too_short", (1, 10), DType.FLOAT32, data=b"")) == 0
        assert (
            gen.sample(Tensor("l_ok", (1, 2), DType.FLOAT32, data=struct.pack("<2f", 2.0, 1.0)))
            == 0
        )
        assert gen.is_eos(2) is False

        # search.py: SearchAlgorithm base select_next_token
        assert SearchAlgorithm().select_next_token(t_short_logits, []) == 0

        # logit_processors.py: short logits (offset < 0 so start < 0 for i=0 and start >= 0 for i=9) and out-of-range token IDs
        t_short_2 = Tensor("sl2", (1, 10), DType.FLOAT32, data=struct.pack("<2f", 1.0, 2.0))
        TopKLogitProcessor(1)([0], t_short_2)
        MinPLogitProcessor(0.5)([0], t_short_logits)
        RepetitionPenaltyLogitProcessor(1.5)([0, 9, 99], t_short_logits)
        # Also test RepetitionPenalty with negative logit at index 9
        t_short_neg = Tensor("sn", (1, 10), DType.FLOAT32, data=struct.pack("<f", -1.0))
        RepetitionPenaltyLogitProcessor(1.5)([0, 9, 99], t_short_neg)
        FrequencyPenaltyLogitProcessor(1.0)([0, 9, 99], t_short_logits)
        PresencePenaltyLogitProcessor(1.0)([0, 9, 99], t_short_logits)

        # NoRepeatNGramLogitProcessor when prefix matches and banned_token >= vocab_size or start < 0
        NoRepeatNGramLogitProcessor(2)([1, 99, 1], t_short_logits)
        NoRepeatNGramLogitProcessor(2)([1, 0, 1], t_short_logits)

        # ForcedBOSLogitProcessor, ForcedEOSLogitProcessor, LogitBiasProcessor, AllowedWordsLogitProcessor with short logits
        ForcedBOSLogitProcessor(9)([], t_short_logits)
        ForcedEOSLogitProcessor(2, 9)([1], t_short_logits)
        LogitBiasProcessor({0: 1.0, 9: 1.0, 99: 1.0})([1], t_short_logits)
        AllowedWordsLogitProcessor([9])([1], t_short_logits)
        NoBadWordsLogitProcessor([[], [0], [1, 9], [1, 99], [2, 3]])([1], t_short_logits)

        # state.py:
        # - PagedKVCache.update twice on same layer_idx, and SequenceBatchingKVCache.update twice on same layer_idx
        pa = PagedKVCache(page_size=2)
        t_kv = Tensor("kv", (1, 2, 1, 4), DType.FLOAT32, data=bytes(32))
        pa.update(t_kv, t_kv, 0)
        pa.update(t_kv, t_kv, 0)
        cbkv = SequenceBatchingKVCache()
        cbkv.update(t_kv, t_kv, 0)
        cbkv.update(t_kv, t_kv, 0)
        # - PositionalEmbeddingUtils.apply_rope and apply_alibi when offset + itemsize > len(new_data)
        t_alibi_short = Tensor("as", (1, 2, 2, 2), DType.FLOAT32, data=struct.pack("<f", 1.0))
        PositionalEmbeddingUtils.apply_rope(t_alibi_short, t_alibi_short, seq_len=2)
        PositionalEmbeddingUtils.apply_alibi(t_alibi_short, num_heads=2)

        # tokenizer.py: decode with clean_up_tokenization_spaces=False on Tokenizer, BPE, WordPiece, Unigram
        ct = Tokenizer()
        assert ct.decode([65, 66], clean_up_tokenization_spaces=False) == "AB"
        bpe = BPETokenizer(vocab={"a": 0}, merges=[])
        assert bpe.decode([0], clean_up_tokenization_spaces=False) == "a"
        wp = WordPieceTokenizer(vocab={"a": 0}, unk_token="[UNK]")
        assert wp.decode([0], clean_up_tokenization_spaces=False) == "a"
        ug = UnigramTokenizer(vocab={"a": (0, -1.0)}, unk_token="<unk>")
        assert ug.decode([0], clean_up_tokenization_spaces=False) == "a"

    asyncio.run(_run())


def test_sphinx_demo_remaining_lines_and_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover remaining lines and branches in onnx9000-sphinx-demo."""
    import runpy
    import sys

    import onnx9000.sphinx_demo as sd_mod

    # Execute module with mock docutils and sphinx in sys.modules to cover try blocks (lines 17, 55) without mutating sd_mod
    monkeypatch.setitem(
        sys.modules,
        "docutils",
        types.SimpleNamespace(
            nodes=types.SimpleNamespace(
                General=type("G", (dict,), {}),
                Element=type("E", (), {}),
                Node=type("N", (dict,), {}),
            )
        ),
    )  # type: ignore[arg-type]
    monkeypatch.setitem(
        sys.modules,
        "docutils.parsers.rst",
        types.SimpleNamespace(Directive=type("D", (), {})),
    )  # type: ignore[arg-type]
    monkeypatch.setitem(
        sys.modules,
        "sphinx.application",
        types.SimpleNamespace(Sphinx=type("S", (), {})),
    )  # type: ignore[arg-type]
    runpy.run_path(str(sd_mod.__file__))
    monkeypatch.undo()

    # copy_frontend_assets when out_static_dir does NOT exist and pyodide_src does NOT exist
    srcdir = tmp_path / "docs"
    srcdir.mkdir()
    dist_dir = tmp_path / "apps" / "sphinx-demo-ui" / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "bundle.js").write_text("// bundle")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    app = types.SimpleNamespace(
        srcdir=str(srcdir),
        outdir=str(out_dir),
        builder=types.SimpleNamespace(format="html"),
        add_node=lambda *_a, **_kw: None,
        add_js_file=lambda *_a, **_kw: None,
        add_css_file=lambda *_a, **_kw: None,
        add_directive=lambda *_a, **_kw: None,
        connect=lambda *_a, **_kw: None,
    )
    copy_frontend_assets(app, None)
    assert (out_dir / "_static" / "demo-ui" / "bundle.js").exists()
    setup(app)  # type: ignore[arg-type]
