"""Tests for packages/python/onnx9000-core/tests/genai/test_logit_processors.py."""

import struct

import pytest
from onnx9000.core.ir import Tensor
from onnx9000.genai.logit_processor_list import LogitProcessorList
from onnx9000.genai.logit_processors import (
    RepetitionPenaltyLogitProcessor,
    TemperatureLogitProcessor,
    TopKLogitProcessor,
)
from onnx9000.genai.top_p import TopPLogitProcessor


def create_logits(vals):
    """Perform create logits operation."""
    data = bytearray(len(vals) * 4)
    for i, v in enumerate(vals):
        data[i * 4 : (i + 1) * 4] = struct.pack("<f", v)
    return Tensor(
        name="logits", shape=(1, len(vals)), data=data, dtype=type("mock", (), {"itemsize": 4})
    )


def extract_logits(tensor):
    """Perform extract logits operation."""
    vals = []
    for i in range(len(tensor.data) // 4):
        vals.append(struct.unpack("<f", tensor.data[i * 4 : (i + 1) * 4])[0])
    return vals


def test_temperature():
    """Test temperature."""
    proc = TemperatureLogitProcessor(2.0)
    logits = create_logits([2.0, 4.0])
    out = proc([], logits)
    vals = extract_logits(out)
    assert vals[0] == 1.0
    assert vals[1] == 2.0


def test_top_k():
    """Test top k."""
    proc = TopKLogitProcessor(2)
    logits = create_logits([1.0, 5.0, 3.0, 2.0])
    out = proc([], logits)
    vals = extract_logits(out)
    assert vals[0] == float("-inf")
    assert vals[1] == 5.0
    assert vals[2] == 3.0
    assert vals[3] == float("-inf")


def test_repetition_penalty():
    """Test repetition penalty."""
    proc = RepetitionPenaltyLogitProcessor(2.0)
    logits = create_logits([1.0, -1.0, 3.0])
    out = proc([1], logits)
    vals = extract_logits(out)
    assert vals[0] == 1.0
    assert vals[1] == -2.0
    assert vals[2] == 3.0


def test_top_p():
    """Test top p."""
    proc = TopPLogitProcessor(0.9)
    logits = create_logits([1.0, 9.0, 10.0])
    out = proc([], logits)
    vals = extract_logits(out)
    assert vals[0] == float("-inf")
    assert vals[1] == 9.0
    assert vals[2] == 10.0


def test_logit_processor_list():
    """Test logit processor list."""
    lst = LogitProcessorList([TemperatureLogitProcessor(2.0), TopKLogitProcessor(1)])
    logits = create_logits([2.0, 4.0])
    out = lst([], logits)
    vals = extract_logits(out)
    assert vals[0] == float("-inf")
    assert vals[1] == 2.0


from onnx9000.genai.logit_processors import (
    FrequencyPenaltyLogitProcessor,
    MinPLogitProcessor,
    PresencePenaltyLogitProcessor,
)


def test_min_p():
    """Test min p."""
    proc = MinPLogitProcessor(0.1)

    logits = create_logits([1.0, 8.0, 10.0])
    out = proc([], logits)
    vals = extract_logits(out)
    assert vals[0] == float("-inf")
    assert vals[1] == 8.0
    assert vals[2] == 10.0


def test_presence_penalty():
    """Test presence penalty."""
    proc = PresencePenaltyLogitProcessor(1.5)
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc([1, 1], logits)
    vals = extract_logits(out)
    assert vals[0] == 2.0
    assert vals[1] == 2.5
    assert vals[2] == 6.0


def test_frequency_penalty():
    """Test frequency penalty."""
    proc = FrequencyPenaltyLogitProcessor(1.5)
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc([1, 1], logits)
    vals = extract_logits(out)
    assert vals[0] == 2.0
    assert vals[1] == 1.0
    assert vals[2] == 6.0


from onnx9000.genai.logit_processors import (
    AllowedWordsLogitProcessor,
    ForcedBOSLogitProcessor,
    ForcedEOSLogitProcessor,
    LogitBiasProcessor,
    NoBadWordsLogitProcessor,
    NoRepeatNGramLogitProcessor,
)


def test_forced_bos():
    """Test forced bos."""
    proc = ForcedBOSLogitProcessor(1)
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc([], logits)
    vals = extract_logits(out)
    assert vals[0] == float("-inf")
    assert vals[1] == 4.0
    assert vals[2] == float("-inf")


def test_no_repeat_ngram():
    """Test no repeat ngram."""
    proc = NoRepeatNGramLogitProcessor(2)
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc([0, 1, 0], logits)
    vals = extract_logits(out)
    assert vals[0] == 2.0
    assert vals[1] == float("-inf")
    assert vals[2] == 6.0


def test_no_bad_words():
    """Test no bad words."""
    proc = NoBadWordsLogitProcessor([[1, 2]])
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc([1], logits)
    vals = extract_logits(out)
    assert vals[0] == 2.0
    assert vals[1] == 4.0
    assert vals[2] == float("-inf")


def test_allowed_words():
    """Test allowed words."""
    proc = AllowedWordsLogitProcessor([0, 2])
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc([], logits)
    vals = extract_logits(out)
    assert vals[0] == 2.0
    assert vals[1] == float("-inf")
    assert vals[2] == 6.0


def test_top_p_errors():
    """Test top p errors."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.top_p import TopPLogitProcessor

    with pytest.raises(ValueError):
        TopPLogitProcessor(-0.5)
    proc = TopPLogitProcessor(1.0)
    t = create_logits([1.0])
    assert proc([], t) is t
    proc2 = TopPLogitProcessor(0.5)
    t_none = Tensor(name="x", shape=(1, 10), data=None)
    assert proc2([], t_none) is t_none


def test_min_p_errors():
    """Test min p errors."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import MinPLogitProcessor

    with pytest.raises(ValueError):
        MinPLogitProcessor(1.5)
    proc = MinPLogitProcessor(1.0)
    t = create_logits([1.0])
    assert proc([], t) is t
    proc2 = MinPLogitProcessor(0.5)
    t_none = Tensor(name="x", shape=(1, 10), data=None)
    assert proc2([], t_none) is t_none
    t_empty = create_logits([])
    assert proc2([], t_empty) is t_empty


def test_temperature_errors():
    """Test temperature errors."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import TemperatureLogitProcessor

    with pytest.raises(ValueError):
        TemperatureLogitProcessor(0.0)
    proc = TemperatureLogitProcessor(1.0)
    t = create_logits([1.0])
    assert proc([], t) is t
    proc2 = TemperatureLogitProcessor(0.5)
    t_none = Tensor(name="x", shape=(1, 10), data=None)
    assert proc2([], t_none) is t_none


def test_top_k_errors():
    """Test top k errors."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import TopKLogitProcessor

    with pytest.raises(ValueError):
        TopKLogitProcessor(0)
    proc = TopKLogitProcessor(5)
    t_none = Tensor(name="x", shape=(1, 10), data=None)
    assert proc([], t_none) is t_none


def test_repetition_penalty_errors():
    """Test repetition penalty errors."""
    from onnx9000.genai.logit_processors import RepetitionPenaltyLogitProcessor

    with pytest.raises(ValueError):
        RepetitionPenaltyLogitProcessor(-1.0)
    proc = RepetitionPenaltyLogitProcessor(1.0)
    t = create_logits([1.0])
    assert proc([], t) is t


def test_presence_frequency_errors():
    """Test presence frequency errors."""
    from onnx9000.genai.logit_processors import (
        FrequencyPenaltyLogitProcessor,
        PresencePenaltyLogitProcessor,
    )

    proc = PresencePenaltyLogitProcessor(0.0)
    t = create_logits([1.0])
    assert proc([], t) is t
    proc2 = FrequencyPenaltyLogitProcessor(0.0)
    assert proc2([], t) is t


def test_forced_bos_eos_errors():
    """Test forced bos eos errors."""
    from onnx9000.genai.logit_processors import ForcedBOSLogitProcessor

    proc = ForcedBOSLogitProcessor(1)
    t = create_logits([1.0])
    assert proc([1], t) is t
    proc2 = ForcedEOSLogitProcessor(2, 1)
    assert proc2([], t) is t


def test_logit_bias_errors():
    """Test logit bias errors."""
    proc = LogitBiasProcessor({})
    t = create_logits([1.0])
    assert proc([], t) is t


def test_no_repeat_ngram_errors():
    """Test no repeat ngram errors."""
    from onnx9000.genai.logit_processors import NoRepeatNGramLogitProcessor

    with pytest.raises(ValueError):
        NoRepeatNGramLogitProcessor(0)
    proc = NoRepeatNGramLogitProcessor(2)
    t = create_logits([1.0])
    assert proc([], t) is t
    assert proc([1], t) is t


def test_no_bad_words_errors():
    """Test no bad words errors."""
    from onnx9000.genai.logit_processors import NoBadWordsLogitProcessor

    proc = NoBadWordsLogitProcessor([])
    t = create_logits([1.0])
    assert proc([], t) is t
    proc2 = NoBadWordsLogitProcessor([[1, 2]])
    assert proc2([], t) is t


def test_allowed_words_errors():
    """Test allowed words errors."""
    from onnx9000.genai.logit_processors import AllowedWordsLogitProcessor

    proc = AllowedWordsLogitProcessor([])
    t = create_logits([1.0])
    assert proc([], t) is t


def test_logit_processors_missing_2():
    """Test logit processors missing 2."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import TemperatureLogitProcessor, TopKLogitProcessor

    t_short = Tensor(
        name="x", shape=(1, 2), data=bytearray(4), dtype=type("mock", (), {"itemsize": 4})
    )
    proc = TemperatureLogitProcessor(0.5)
    proc([], t_short)
    proc = TopKLogitProcessor(1)
    proc([], t_short)
    from onnx9000.genai.top_p import TopPLogitProcessor

    proc = TopPLogitProcessor(0.5)
    proc([], t_short)


def test_logit_processors_more():
    """Test logit processors more."""
    import struct

    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import (
        RepetitionPenaltyLogitProcessor,
    )

    p = RepetitionPenaltyLogitProcessor(penalty=2.0)
    data = bytearray(8)
    data[0:4] = struct.pack("<f", 4.0)
    data[4:8] = struct.pack("<f", -4.0)
    t = Tensor(name="", shape=(2,), data=data, dtype=type("m", (), {"itemsize": 4}))
    p([0], t)
    eos = ForcedEOSLogitProcessor(max_length=1, eos_token_id=0)
    data2 = bytearray(8)
    data2[0:4] = struct.pack("<f", 0.0)
    data2[4:8] = struct.pack("<f", 0.0)
    t2 = Tensor(name="", shape=(2,), data=data2, dtype=type("m", (), {"itemsize": 4}))
    eos([1], t2)


def test_forced_eos_no_force():
    """Test forced eos no force."""
    from onnx9000.core.ir import Tensor

    eos = ForcedEOSLogitProcessor(max_length=5, eos_token_id=0)
    t = Tensor(name="", shape=(2,), data=bytearray(8), dtype=type("m", (), {"itemsize": 4}))
    assert eos([1], t) is t


def test_forced_eos_force():
    """Test forced eos force."""
    from onnx9000.core.ir import Tensor

    eos = ForcedEOSLogitProcessor(max_length=1, eos_token_id=0)
    t = Tensor(name="", shape=(2,), data=bytearray(8), dtype=type("m", (), {"itemsize": 4}))
    eos([], t)


def test_logit_processors_11_287():
    """Test logit processors 11 287."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import (
        AllowedWordsLogitProcessor,
        LogitProcessor,
        NoBadWordsLogitProcessor,
    )

    p = LogitProcessor()
    t = Tensor(name="", shape=(), data=bytearray(4))
    assert p(None, t) is t
    bad = NoBadWordsLogitProcessor([[1]])
    bad([0], t)
    allowed = AllowedWordsLogitProcessor([1])
    allowed([0], t)


def test_logit_bias_processor_restored():
    """Test logit bias processor restored."""
    import struct

    from onnx9000.core.ir import Tensor

    p = LogitBiasProcessor(bias_map={0: 1.0, 1: -1.0})
    data = bytearray(8)
    data[0:4] = struct.pack("<f", 0.0)
    data[4:8] = struct.pack("<f", 0.0)
    t = Tensor(name="", shape=(2,), data=data, dtype=type("m", (), {"itemsize": 4}))
    p_empty = LogitBiasProcessor(bias_map={})
    assert p_empty([0], t) is t
    res = p([0], t)
    assert len(res.data) == 8


def test_chunking_restored():
    """Test chunking restored."""
    import onnx9000.core.ir as ir
    from onnx9000.genai.chunking import ChunkManager

    c = ChunkManager.chunk_model("a.onnx")
    ChunkManager.create_manifest(c, "/tmp/")
    m = ir.Graph(name="test")
    ChunkManager.externalize_weights(m, "/tmp/out")
    ChunkManager.embed_tokenizer(m, {})
