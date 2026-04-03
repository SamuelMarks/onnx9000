"""Tests for frontend weight_utils."""

import numpy as np
import pytest

from unittest.mock import MagicMock
from onnx9000.converters.frontend.weight_utils import export_state_dict, universal_weight_bridge
from onnx9000.core.ir import Graph, Tensor, Constant
from onnx9000.core.dtypes import DType


def test_export_state_dict():
    graph = Graph(name="mygraph")
    t1 = Constant(
        name="a/b",
        shape=(2,),
        dtype=DType.FLOAT32,
        values=np.array([1.0, 2.0], dtype=np.float32).tobytes(),
    )
    t2 = Tensor(name="t2", shape=(1,), dtype=DType.FLOAT32)  # not a constant
    t3 = Constant(
        name="c", shape=(1,), dtype=DType.INT64, values=np.array([42], dtype=np.int64).tobytes()
    )

    graph.tensors = {"a/b": t1, "t2": t2, "c": t3}

    state_dict = export_state_dict(graph)

    assert "a.b" in state_dict
    assert "c" in state_dict
    assert "t2" not in state_dict

    try:
        import torch

        assert isinstance(state_dict["a.b"], torch.Tensor)
        assert isinstance(state_dict["c"], torch.Tensor)
        assert state_dict["a.b"].tolist() == [1.0, 2.0]
        assert state_dict["c"].tolist() == [42]
    except ImportError:
        assert isinstance(state_dict["a.b"], np.ndarray)


def test_universal_weight_bridge():
    weights = {"w": np.array([1.0])}

    # pytorch
    assert universal_weight_bridge(weights, "pytorch") == weights

    # other
    assert universal_weight_bridge(weights, "other") == weights

    # safetensors
    import sys

    with pytest.MonkeyPatch.context() as m:
        m.setitem(sys.modules, "safetensors", None)
        m.setitem(sys.modules, "safetensors.torch", None)
        res = universal_weight_bridge(weights, "safetensors")
        assert res is None

    with pytest.MonkeyPatch.context() as m:
        mock_save = MagicMock(return_value=b"saved")
        mock_st = MagicMock()
        mock_st.save = mock_save
        m.setitem(sys.modules, "safetensors.torch", mock_st)

        # Test success branch
        import types

        mock_st_module = types.ModuleType("safetensors")
        mock_st_torch_module = types.ModuleType("safetensors.torch")
        mock_st_torch_module.save = mock_save
        m.setitem(sys.modules, "safetensors", mock_st_module)
        m.setitem(sys.modules, "safetensors.torch", mock_st_torch_module)

        res = universal_weight_bridge(weights, "safetensors")
        assert res == b"saved"


def test_export_state_dict_import_error():
    import sys

    with pytest.MonkeyPatch.context() as m:
        m.setitem(sys.modules, "torch", None)

        graph = Graph(name="mygraph")
        t1 = Constant(
            name="a/b",
            shape=(2,),
            dtype=DType.FLOAT32,
            values=np.array([1.0, 2.0], dtype=np.float32).tobytes(),
        )
        graph.tensors = {"a/b": t1}

        state_dict = export_state_dict(graph)
        assert isinstance(state_dict["a.b"], np.ndarray)
