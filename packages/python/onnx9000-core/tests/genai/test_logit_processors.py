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
    data = bytearray(len(vals) * 4)
    for i, v in enumerate(vals):
        data[i * 4 : (i + 1) * 4] = struct.pack("<f", v)

    # Just a mock shape and dtype
    return Tensor(
        name="logits",
        shape=(1, len(vals)),
        data=data,
        dtype=type("mock", (), {"itemsize": 4}),
    )


def extract_logits(tensor):
    vals = []
    for i in range(len(tensor.data) // 4):
        vals.append(struct.unpack("<f", tensor.data[i * 4 : (i + 1) * 4])[0])
    return vals


def test_temperature():
    proc = TemperatureLogitProcessor(2.0)
    logits = create_logits([2.0, 4.0])
    out = proc([], logits)

    vals = extract_logits(out)
    assert vals[0] == 1.0
    assert vals[1] == 2.0


def test_top_k():
    proc = TopKLogitProcessor(2)
    logits = create_logits([1.0, 5.0, 3.0, 2.0])
    out = proc([], logits)

    vals = extract_logits(out)
    assert vals[0] == float("-inf")
    assert vals[1] == 5.0
    assert vals[2] == 3.0
    assert vals[3] == float("-inf")


def test_repetition_penalty():
    proc = RepetitionPenaltyLogitProcessor(2.0)
    logits = create_logits([1.0, -1.0, 3.0])
    out = proc([1], logits)  # Token 1 was generated

    vals = extract_logits(out)
    assert vals[0] == 1.0
    assert vals[1] == -2.0  # -1.0 * 2.0
    assert vals[2] == 3.0


def test_top_p():
    proc = TopPLogitProcessor(0.9)
    # create somewhat sharp probabilities so we know what is dropped
    # e.g., 10, 9, 1 => probs are ~0.73, ~0.27, ~0.0001
    logits = create_logits([1.0, 9.0, 10.0])
    out = proc([], logits)

    vals = extract_logits(out)
    assert vals[0] == float("-inf")  # 1.0 gets dropped because 10+9 is 1.0
    assert vals[1] == 9.0
    assert vals[2] == 10.0


def test_logit_processor_list():
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
    proc = MinPLogitProcessor(0.1)
    import math

    # max logit is 10.0
    # threshold = 10.0 + log(0.1) = 10.0 - 2.302 = 7.698
    logits = create_logits([1.0, 8.0, 10.0])
    out = proc([], logits)

    vals = extract_logits(out)
    assert vals[0] == float("-inf")
    assert vals[1] == 8.0
    assert vals[2] == 10.0


def test_presence_penalty():
    proc = PresencePenaltyLogitProcessor(1.5)
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc([1, 1], logits)  # Token 1 was generated twice

    vals = extract_logits(out)
    assert vals[0] == 2.0
    assert vals[1] == 2.5  # 4.0 - 1.5
    assert vals[2] == 6.0


def test_frequency_penalty():
    proc = FrequencyPenaltyLogitProcessor(1.5)
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc([1, 1], logits)  # Token 1 was generated twice

    vals = extract_logits(out)
    assert vals[0] == 2.0
    assert vals[1] == 1.0  # 4.0 - 1.5 * 2
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
    proc = ForcedBOSLogitProcessor(1)
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc([], logits)
    vals = extract_logits(out)
    assert vals[0] == float("-inf")
    assert vals[1] == 4.0
    assert vals[2] == float("-inf")


def test_no_repeat_ngram():
    proc = NoRepeatNGramLogitProcessor(2)
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc(
        [0, 1, 0], logits
    )  # the bigram (0, 1) should be banned next, wait... prefix is [0]. where did [0] appear? index 0. So next is 1. token 1 is banned.
    vals = extract_logits(out)
    assert vals[0] == 2.0
    assert vals[1] == float("-inf")
    assert vals[2] == 6.0


def test_no_bad_words():
    proc = NoBadWordsLogitProcessor([[1, 2]])
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc([1], logits)  # next token 2 is banned
    vals = extract_logits(out)
    assert vals[0] == 2.0
    assert vals[1] == 4.0
    assert vals[2] == float("-inf")


def test_allowed_words():
    proc = AllowedWordsLogitProcessor([0, 2])
    logits = create_logits([2.0, 4.0, 6.0])
    out = proc([], logits)
    vals = extract_logits(out)
    assert vals[0] == 2.0
    assert vals[1] == float("-inf")
    assert vals[2] == 6.0


def test_top_p_errors():
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.top_p import TopPLogitProcessor

    with pytest.raises(ValueError):
        TopPLogitProcessor(-0.5)

    proc = TopPLogitProcessor(1.0)  # top_p >= 1.0 returns logits
    t = create_logits([1.0])
    assert proc([], t) is t

    proc2 = TopPLogitProcessor(0.5)
    t_none = Tensor(name="x", shape=(1, 10), data=None)
    assert proc2([], t_none) is t_none


def test_min_p_errors():
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

    # max_val = float('-inf')
    t_empty = create_logits([])
    assert proc2([], t_empty) is t_empty


def test_temperature_errors():
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
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import TopKLogitProcessor

    with pytest.raises(ValueError):
        TopKLogitProcessor(0)

    proc = TopKLogitProcessor(5)
    t_none = Tensor(name="x", shape=(1, 10), data=None)
    assert proc([], t_none) is t_none


def test_repetition_penalty_errors():
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import RepetitionPenaltyLogitProcessor

    with pytest.raises(ValueError):
        RepetitionPenaltyLogitProcessor(-1.0)

    proc = RepetitionPenaltyLogitProcessor(1.0)
    t = create_logits([1.0])
    assert proc([], t) is t


def test_presence_frequency_errors():
    from onnx9000.core.ir import Tensor
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
    from onnx9000.genai.logit_processors import (
        ForcedBOSLogitProcessor,
        ForcedEOSLogitProcessor,
    )

    proc = ForcedBOSLogitProcessor(1)
    t = create_logits([1.0])
    assert proc([1], t) is t  # length > 0

    proc2 = ForcedEOSLogitProcessor(2, 1)
    assert proc2([], t) is t  # not length == max_length - 1


def test_logit_bias_errors():
    from onnx9000.genai.logit_processors import LogitBiasProcessor

    proc = LogitBiasProcessor({})
    t = create_logits([1.0])
    assert proc([], t) is t


def test_no_repeat_ngram_errors():
    from onnx9000.genai.logit_processors import NoRepeatNGramLogitProcessor

    with pytest.raises(ValueError):
        NoRepeatNGramLogitProcessor(0)

    proc = NoRepeatNGramLogitProcessor(2)
    t = create_logits([1.0])
    assert proc([], t) is t  # input_ids < ngram - 1
    assert proc([1], t) is t  # no banned tokens


def test_no_bad_words_errors():
    from onnx9000.genai.logit_processors import NoBadWordsLogitProcessor

    proc = NoBadWordsLogitProcessor([])
    t = create_logits([1.0])
    assert proc([], t) is t

    proc2 = NoBadWordsLogitProcessor([[1, 2]])
    assert proc2([], t) is t  # prefix not match


def test_allowed_words_errors():
    from onnx9000.genai.logit_processors import AllowedWordsLogitProcessor

    proc = AllowedWordsLogitProcessor([])
    t = create_logits([1.0])
    assert proc([], t) is t


def test_logit_processors_missing_2():
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import (
        TemperatureLogitProcessor,
        TopKLogitProcessor,
    )

    t_short = Tensor(
        name="x",
        shape=(1, 2),
        data=bytearray(4),
        dtype=type("mock", (), {"itemsize": 4}),
    )

    proc = TemperatureLogitProcessor(0.5)
    proc([], t_short)

    proc = TopKLogitProcessor(1)
    proc([], t_short)

    from onnx9000.genai.top_p import TopPLogitProcessor

    proc = TopPLogitProcessor(0.5)
    proc([], t_short)


def test_logit_processors_more():
    import struct

    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import (
        ForcedEOSLogitProcessor,
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
    # len is 1, so it forces eos.


def test_forced_eos_no_force():
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import ForcedEOSLogitProcessor

    eos = ForcedEOSLogitProcessor(max_length=5, eos_token_id=0)
    t = Tensor(name="", shape=(2,), data=bytearray(8), dtype=type("m", (), {"itemsize": 4}))
    assert eos([1], t) is t


def test_forced_eos_force():
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import ForcedEOSLogitProcessor

    eos = ForcedEOSLogitProcessor(max_length=1, eos_token_id=0)
    t = Tensor(name="", shape=(2,), data=bytearray(8), dtype=type("m", (), {"itemsize": 4}))
    eos([], t)


def test_logit_processors_11_287():
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
    import struct

    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import LogitBiasProcessor

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
    import onnx9000.core.ir as ir
    from onnx9000.genai.chunking import ChunkManager

    c = ChunkManager.chunk_model("a.onnx")
    ChunkManager.create_manifest(c, "/tmp/")

    m = ir.Graph(name="test")
    ChunkManager.externalize_weights(m, "/tmp/out")
    ChunkManager.embed_tokenizer(m, {})
