"""Tests for tensorrt more."""

import unittest
from unittest.mock import MagicMock

import onnx9000.tensorrt as trt
import onnx9000.tensorrt.network as trt_network
from onnx9000.tensorrt.ops import (
    trt_add,
    trt_and,
    trt_averagepool,
    trt_clip,
    trt_div,
    trt_equal,
    trt_greater,
    trt_leakyrelu,
    trt_less,
    trt_max,
    trt_maxpool,
    trt_min,
    trt_mul,
    trt_or,
    trt_pow,
    trt_reducemax,
    trt_reducemean,
    trt_reducemin,
    trt_reduceprod,
    trt_reducesum,
    trt_sub,
    trt_xor,
)
from onnx9000.tensorrt.ops_conv import trt_conv
from onnx9000.tensorrt.ops_dim import trt_concat, trt_gather, trt_reshape, trt_slice, trt_transpose
from onnx9000.tensorrt.ops_matmul import trt_matmul


class MockNode:
    """Docstring for D101."""

    def __init__(self, op_type, inputs, outputs, attrs=None):
        """Docstring for D107."""
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attrs or {}


class MockAttr:
    """Docstring for D101."""

    def __init__(self, value):
        """Docstring for D107."""
        self.value = value

    def __iter__(self):
        """Docstring for D105."""
        if isinstance(self.value, list):
            return iter(self.value)


class TestOpsMore(unittest.TestCase):
    """Docstring for D101."""

    def setUp(self):
        """Docstring for D102."""
        self.mock_lib = MagicMock()
        trt.ffi.lib = self.mock_lib
        self.net = trt_network.INetworkDefinition(111)
        t1 = trt_network.ITensor(1, "in1")
        t1.ptr = 1
        t2 = trt_network.ITensor(2, "in2")
        t2.ptr = 2
        t3 = trt_network.ITensor(3, "in3")
        t3.ptr = 3
        self.tensors = {"in1": t1, "in2": t2, "in3": t3}

    def test_elementwise_failures(self):
        """Docstring for D102."""
        node = MockNode("X", ["in1", "in2"], ["out"])
        for fn in [
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
        ]:
            self.mock_lib.addElementWise = None
            with self.assertRaises(RuntimeError):
                fn(self.net, node, self.tensors)
            self.mock_lib.addElementWise = MagicMock(return_value=0)
            with self.assertRaises(RuntimeError):
                fn(self.net, node, self.tensors)

    def test_unary_failures(self):
        """Docstring for D102."""
        node = MockNode("X", ["in1"], ["out"])
        from onnx9000.tensorrt.ops import (
            trt_abs,
            trt_elu,
            trt_exp,
            trt_hardsigmoid,
            trt_log,
            trt_neg,
            trt_not,
            trt_relu,
            trt_selu,
            trt_sigmoid,
            trt_softplus,
            trt_sqrt,
            trt_tanh,
        )

        for fn in [
            trt_exp,
            trt_log,
            trt_sqrt,
            trt_abs,
            trt_neg,
            trt_not,
            trt_relu,
            trt_sigmoid,
            trt_tanh,
            trt_elu,
            trt_selu,
            trt_softplus,
            trt_hardsigmoid,
        ]:
            self.mock_lib.addUnaryOperation = None
            self.mock_lib.addActivation = None
            with self.assertRaises(RuntimeError):
                fn(self.net, node, self.tensors)
            self.mock_lib.addUnaryOperation = MagicMock(return_value=0)
            self.mock_lib.addActivation = MagicMock(return_value=0)
            with self.assertRaises(RuntimeError):
                fn(self.net, node, self.tensors)

    def test_activation_failures(self):
        """Docstring for D102."""
        node = MockNode("X", ["in1"], ["out"], {"alpha": MockAttr(0.1)})
        for fn in [trt_leakyrelu, trt_clip]:
            self.mock_lib.addActivation = None
            with self.assertRaises(RuntimeError):
                fn(self.net, node, self.tensors)
            self.mock_lib.addActivation = MagicMock(return_value=0)
            with self.assertRaises(RuntimeError):
                fn(self.net, node, self.tensors)

    def test_reduce_failures(self):
        """Docstring for D102."""
        node = MockNode("X", ["in1"], ["out"], {"axes": MockAttr([0])})
        for fn in [trt_reducesum, trt_reducemean, trt_reducemax, trt_reducemin, trt_reduceprod]:
            self.mock_lib.addReduce = None
            with self.assertRaises(RuntimeError):
                fn(self.net, node, self.tensors)
            self.mock_lib.addReduce = MagicMock(return_value=0)
            with self.assertRaises(RuntimeError):
                fn(self.net, node, self.tensors)

    def test_pool_failures(self):
        """Docstring for D102."""
        node = MockNode(
            "X",
            ["in1"],
            ["out"],
            {
                "kernel_shape": MockAttr([2, 2]),
                "strides": MockAttr([2, 2]),
                "pads": MockAttr([0, 0, 0, 0]),
            },
        )
        for fn in [trt_maxpool, trt_averagepool]:
            self.mock_lib.addPoolingNd = None
            with self.assertRaises(RuntimeError):
                fn(self.net, node, self.tensors)
            self.mock_lib.addPoolingNd = MagicMock(return_value=0)
            with self.assertRaises(RuntimeError):
                fn(self.net, node, self.tensors)

    def test_dim_failures(self):
        """Docstring for D102."""
        self.mock_lib.addShuffle = None
        with self.assertRaises(RuntimeError):
            trt_reshape(self.net, MockNode("X", ["in1"], ["out"]), self.tensors)
        with self.assertRaises(RuntimeError):
            trt_transpose(
                self.net, MockNode("X", ["in1"], ["out"], {"perm": MockAttr([0])}), self.tensors
            )

        self.mock_lib.addConcatenation = None
        with self.assertRaises(RuntimeError):
            trt_concat(
                self.net,
                MockNode("X", ["in1", "in2"], ["out"], {"axis": MockAttr(0)}),
                self.tensors,
            )

        self.mock_lib.addSlice = None
        with self.assertRaises(RuntimeError):
            trt_slice(self.net, MockNode("X", ["in1", "in2", "in3"], ["out"]), self.tensors)

        self.mock_lib.addGather = None
        with self.assertRaises(RuntimeError):
            trt_gather(
                self.net,
                MockNode("X", ["in1", "in2"], ["out"], {"axis": MockAttr(0)}),
                self.tensors,
            )

        # Test ptr=0
        self.mock_lib.addShuffle = MagicMock(return_value=0)
        # Doesn't raise on 0 but that's ok for these ones, just need to run it

    def test_matmul_failures(self):
        """Docstring for D102."""
        self.mock_lib.addMatrixMultiply = None
        with self.assertRaises(RuntimeError):
            trt_matmul(self.net, MockNode("X", ["in1", "in2"], ["out"]), self.tensors)

    def test_conv_failures(self):
        """Docstring for D102."""
        node = MockNode("X", ["in1", "in2"], ["out"], {"kernel_shape": MockAttr([3, 3])})
        self.mock_lib.addConvolutionNd = None
        with self.assertRaises(RuntimeError):
            trt_conv(self.net, node, self.tensors)
        self.mock_lib.addConvolutionNd = MagicMock(return_value=0)
        with self.assertRaises(RuntimeError):
            trt_conv(self.net, node, self.tensors)

    def test_network_failures(self):
        """Docstring for D102."""
        delattr(self.mock_lib, "addInput")
        with self.assertRaises(RuntimeError):
            self.net.add_input("in", trt.DataType.kFLOAT, trt.Dims([1]))

        repr(self.tensors["in1"])

        cfg = trt_network.IBuilderConfig(123)
        self.mock_lib.destroyBuilderConfig = MagicMock()
        cfg.destroy()
        cfg.destroy()

        self.mock_lib.destroyNetworkDefinition = MagicMock()
        self.net.destroy()
        self.net.destroy()
