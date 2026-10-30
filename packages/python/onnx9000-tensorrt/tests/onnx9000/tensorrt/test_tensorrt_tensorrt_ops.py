"""Comprehensive unit tests for onnx9000.tensorrt.ops."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

from onnx9000.tensorrt.network import INetworkDefinition, ITensor
from onnx9000.tensorrt.ops import (
    _get_input,
    trt_abs,
    trt_add,
    trt_and,
    trt_averagepool,
    trt_clip,
    trt_div,
    trt_elu,
    trt_equal,
    trt_exp,
    trt_greater,
    trt_hardsigmoid,
    trt_leakyrelu,
    trt_less,
    trt_log,
    trt_max,
    trt_maxpool,
    trt_min,
    trt_mul,
    trt_neg,
    trt_not,
    trt_or,
    trt_pow,
    trt_reducemax,
    trt_reducemean,
    trt_reducemin,
    trt_reduceprod,
    trt_reducesum,
    trt_relu,
    trt_selu,
    trt_sigmoid,
    trt_softplus,
    trt_sqrt,
    trt_sub,
    trt_tanh,
    trt_xor,
)


@dataclass
class _StubTRTNode:
    """Typed stub node for TensorRT op unit tests."""

    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, int | float | bool | list[int]] = field(default_factory=dict)


def test_all_trt_ops_success() -> None:
    """Verify every TensorRT operator registers an output ITensor when FFI returns a valid pointer."""
    mock_ffi = MagicMock()
    mock_ffi.lib.addElementWise.return_value = 900
    mock_ffi.lib.addUnaryOperation.return_value = 901
    mock_ffi.lib.addActivation.return_value = 902
    mock_ffi.lib.addPoolingNd.return_value = 903
    mock_ffi.lib.addReduce.return_value = 904

    net = INetworkDefinition(100)
    t1 = ITensor(10, "in1")
    t2 = ITensor(20, "in2")
    tensors: dict[str, ITensor] = {"in1": t1, "in2": t2}

    bin_node = _StubTRTNode(["in1", "in2"], ["out_bin"])
    un_node = _StubTRTNode(["in1"], ["out_un"])
    red_node = _StubTRTNode(
        ["in1"],
        ["out_red"],
        attributes={"keepdims": 0, "axes": [0, 2]},
    )
    red_scalar_axes = _StubTRTNode(["in1"], ["out_red2"], attributes={"axes": 1})

    assert _get_input(bin_node, tensors, 0) is t1
    assert _get_input(bin_node, tensors, 5) is None

    with patch("onnx9000.tensorrt.ops.ffi", mock_ffi):
        for fn in (
            trt_add,
            trt_sub,
            trt_mul,
            trt_div,
            trt_max,
            trt_min,
            trt_pow,
            trt_equal,
            trt_less,
            trt_greater,
            trt_and,
            trt_or,
            trt_xor,
        ):
            fn(net, bin_node, tensors)
            assert tensors["out_bin"].ptr == 900

        for fn in (trt_exp, trt_log, trt_sqrt, trt_abs, trt_neg, trt_not):
            fn(net, un_node, tensors)
            assert tensors["out_un"].ptr == 901

        for fn in (
            trt_relu,
            trt_sigmoid,
            trt_tanh,
            trt_leakyrelu,
            trt_elu,
            trt_selu,
            trt_softplus,
            trt_clip,
            trt_hardsigmoid,
        ):
            fn(net, un_node, tensors)
            assert tensors["out_un"].ptr == 902

        for fn in (trt_maxpool, trt_averagepool):
            fn(net, un_node, tensors)
            assert tensors["out_un"].ptr == 903

        for fn in (
            trt_reducemean,
            trt_reducesum,
            trt_reducemax,
            trt_reducemin,
            trt_reduceprod,
        ):
            fn(net, red_node, tensors)
            assert tensors["out_red"].ptr == 904
        trt_reducemean(net, red_scalar_axes, tensors)
        assert tensors["out_red2"].ptr == 904
        trt_reducemean(net, un_node, tensors)
        assert tensors["out_un"].ptr == 904
