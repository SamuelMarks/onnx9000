"""Unit tests for logit processors in `onnx9000.genai.logit_processors`."""

import math
import struct

from onnx9000.core.ir import Tensor
from onnx9000.genai.logit_processors import (
    AllowedWordsLogitProcessor,
    ForcedBOSLogitProcessor,
    ForcedEOSLogitProcessor,
    FrequencyPenaltyLogitProcessor,
    LogitBiasProcessor,
    LogitProcessor,
    MinPLogitProcessor,
    NoBadWordsLogitProcessor,
    NoRepeatNGramLogitProcessor,
    PresencePenaltyLogitProcessor,
    RepetitionPenaltyLogitProcessor,
    TemperatureLogitProcessor,
    TopKLogitProcessor,
)


def _make_logits(vals: list[float]) -> Tensor:
    """Create a 1D float32 logits Tensor."""
    return Tensor(
        name="logits", shape=(1, len(vals)), data=bytearray(struct.pack(f"<{len(vals)}f", *vals))
    )


def _read_logits(t: Tensor) -> list[float]:
    """Unpack float32 values from a logits Tensor."""
    count = len(t.data) // 4
    return list(struct.unpack(f"<{count}f", t.data))


def test_all_logit_processors() -> None:
    """Verify all LogitProcessor implementations modify logits as expected."""
    base = LogitProcessor()
    t0 = _make_logits([1.0, 2.0])
    assert base([], t0) is t0

    temp_out = _read_logits(TemperatureLogitProcessor(2.0)([], _make_logits([2.0, 4.0])))
    assert temp_out == [1.0, 2.0]

    topk_out = _read_logits(TopKLogitProcessor(1)([], _make_logits([1.0, 5.0])))
    assert math.isinf(topk_out[0]) and topk_out[1] == 5.0

    rep_out = _read_logits(RepetitionPenaltyLogitProcessor(2.0)([0], _make_logits([4.0, 2.0])))
    assert rep_out == [2.0, 2.0]

    minp_out = _read_logits(MinPLogitProcessor(0.5)([], _make_logits([0.0, 10.0])))
    assert math.isinf(minp_out[0]) and minp_out[1] == 10.0

    pres_out = _read_logits(PresencePenaltyLogitProcessor(1.5)([0, 0], _make_logits([3.0, 2.0])))
    assert pres_out == [1.5, 2.0]

    freq_out = _read_logits(FrequencyPenaltyLogitProcessor(1.5)([0, 0], _make_logits([4.0, 2.0])))
    assert freq_out == [1.0, 2.0]

    bos_out = _read_logits(ForcedBOSLogitProcessor(1)([], _make_logits([1.0, 2.0])))
    assert math.isinf(bos_out[0]) and bos_out[1] == 2.0

    eos_out = _read_logits(
        ForcedEOSLogitProcessor(max_length=2, eos_token_id=1)([0], _make_logits([1.0, 2.0]))
    )
    assert math.isinf(eos_out[0])

    bias_out = _read_logits(LogitBiasProcessor({0: 2.5})([], _make_logits([1.0, 2.0])))
    assert bias_out == [3.5, 2.0]

    ngram_out = _read_logits(NoRepeatNGramLogitProcessor(2)([0, 1, 0], _make_logits([1.0, 2.0])))
    assert math.isinf(ngram_out[1])

    bad_out = _read_logits(NoBadWordsLogitProcessor([[0, 1]])([0], _make_logits([1.0, 2.0])))
    assert math.isinf(bad_out[1])

    allow_out = _read_logits(AllowedWordsLogitProcessor([0])([], _make_logits([1.0, 2.0])))
    assert allow_out[0] == 1.0 and math.isinf(allow_out[1])
