import pytest
from onnx9000.genai.state import ContinuousKVCache
from onnx9000.core.ir import Tensor
from onnx9000.core.dtypes import DType


def test_kvstate_update_existing():
    state = ContinuousKVCache()
    t1 = Tensor("k1", DType.FLOAT32, (1,))
    t2 = Tensor("v1", DType.FLOAT32, (1,))
    state.update(t1, t2, 0)

    t3 = Tensor("k2", DType.FLOAT32, (1,))
    t4 = Tensor("v2", DType.FLOAT32, (1,))
    state.update(t3, t4, 0)  # triggers pass

    assert state.get(0) == (t3, t4)
