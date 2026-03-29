import unittest
from unittest.mock import MagicMock, patch
import onnx9000.tensorrt as trt
import onnx9000.tensorrt.ffi as trt_ffi
import onnx9000.tensorrt.builder as trt_builder
import onnx9000.tensorrt.network as trt_network
import onnx9000.tensorrt.ops as trt_ops
import onnx9000.tensorrt.ops_conv as trt_ops_conv
import onnx9000.tensorrt.ops_dim as trt_ops_dim
import onnx9000.tensorrt.ops_matmul as trt_ops_matmul


class TestTensorRTFFI(unittest.TestCase):
    def setUp(self):
        # Mock FFI lib
        self.mock_lib = MagicMock()
        trt.ffi.lib = self.mock_lib
        trt.ffi.plugin_lib = self.mock_lib

        trt.ffi._setup_logging_callback()

    def test_ffi_initialization(self):
        self.assertIsNotNone(trt.ffi)
        self.assertIsInstance(trt.ffi.pointers, dict)
        trt.ffi.register_pointer(123, "obj")
        self.assertEqual(trt.ffi.pointers[123], "obj")
        trt.ffi.unregister_pointer(123)
        self.assertNotIn(123, trt.ffi.pointers)

        with self.assertRaises(RuntimeError):
            trt.ffi.check_error(1, "Failed")
        trt.ffi.check_error(0, "OK")

        obj = trt.ffi
        obj._c_log_callback("user_data", 0, b"internal error")
        obj._c_log_callback("user_data", 1, b"error")
        obj._c_log_callback("user_data", 2, b"warning")
        obj._c_log_callback("user_data", 3, b"info")
        obj._c_log_callback("user_data", 4, b"verbose")

    def test_enums(self):
        self.assertEqual(trt.DataType.kFLOAT.value, 0)
        self.assertEqual(trt.ElementWiseOperation.kSUM.value, 0)

    def test_structs(self):
        with self.assertRaises(ValueError):
            trt.Dims([1, 2, 3, 4, 5, 6, 7, 8, 9])
        dims = trt.Dims([1, 2, 3])
        self.assertEqual(dims.nbDims, 3)
        self.assertEqual(dims.d[0], 1)
        self.assertEqual(dims.d[1], 2)
        self.assertEqual(dims.d[2], 3)

        weights = trt.Weights()
        weights.type = trt.DataType.kFLOAT.value
        weights.count = 100
        self.assertEqual(weights.type, 0)
        self.assertEqual(weights.count, 100)

    def test_registry(self):

        trt.register_op("", "Dummy")(lambda x, y, z: True)
        trt.register_op("", "Dummy")(lambda x, y, z: True)
        func = trt.get_op_translator("", "Dummy")
        self.assertIsNotNone(func)
        with self.assertRaises(RuntimeError):
            trt.get_op_translator("", "NonExistentDummy")

    def test_builder(self):
        self.mock_lib.createInferBuilder_INTERNAL.return_value = 1234
        self.mock_lib.createNetworkV2.return_value = 5678
        self.mock_lib.createBuilderConfig.return_value = 9012

        builder = trt_builder.Builder()
        self.assertEqual(builder.ptr, 1234)

        net = builder.create_network()
        self.assertIsInstance(net, trt_builder.NetworkDefinition)
        self.assertEqual(net.ptr, 5678)

        cfg = builder.create_builder_config()
        self.assertIsInstance(cfg, trt_builder.BuilderConfig)
        self.assertEqual(cfg.ptr, 9012)

        cfg.set_memory_pool_limit(trt.MemoryPoolType.kWORKSPACE, 1024)
        del self.mock_lib.setMemoryPoolLimit
        cfg.set_memory_pool_limit(trt.MemoryPoolType.kWORKSPACE, 1024)

        net.destroy()
        builder.destroy()

    def test_builder_fail(self):
        trt.ffi.lib = None
        with self.assertRaises(RuntimeError):
            trt_builder.Builder()
        trt.ffi.lib = self.mock_lib
        del self.mock_lib.createInferBuilder_INTERNAL
        with self.assertRaises(RuntimeError):
            trt_builder.Builder()

        self.mock_lib.createInferBuilder_INTERNAL = MagicMock(return_value=0)
        with self.assertRaises(RuntimeError):
            trt_builder.Builder()
        self.mock_lib.createInferBuilder_INTERNAL.return_value = 1234
        builder = trt_builder.Builder()

        self.mock_lib.createNetworkV2 = MagicMock(return_value=0)
        with self.assertRaises(RuntimeError):
            builder.create_network()

        del self.mock_lib.createNetworkV2
        with self.assertRaises(RuntimeError):
            builder.create_network()

        del self.mock_lib.createBuilderConfig
        with self.assertRaises(RuntimeError):
            builder.create_builder_config()

    def test_network(self):
        net = trt_network.INetworkDefinition(111)

        dims = trt.Dims([1])
        t_ptr = 123
        self.mock_lib.addInput.return_value = t_ptr

        # Proper ctypes conversion for add_input
        self.mock_lib.addInput.argtypes = []

        # We need to bypass the TypeError in the source if we pass MagicMock
        with patch("ctypes.pointer", return_value=MagicMock()):
            with patch("ctypes.c_void_p", return_value=MagicMock()):
                t = net.add_input("in", trt.DataType.kFLOAT, dims)
                t.ptr = 123
                self.assertIsInstance(t, trt_network.ITensor)

                self.mock_lib.addInput.return_value = 0
                with self.assertRaises(RuntimeError):
                    net.add_input("in", trt.DataType.kFLOAT, dims)

                del self.mock_lib.addInput
                with self.assertRaises(RuntimeError):
                    net.add_input("in", trt.DataType.kFLOAT, dims)

        net.mark_output(t)
        self.mock_lib.markOutput = MagicMock()
        net.mark_output(t)

        del self.mock_lib.markOutput
        with self.assertRaises(RuntimeError):
            net.mark_output(t)

    def test_ffi_reload(self):
        import sys

        old_platform = sys.platform
        sys.platform = "win32"
        # Since we just want to execute _load_library with a diff platform
        with patch("ctypes.util.find_library", return_value="dummy.dll"):
            with patch("ctypes.CDLL", return_value=MagicMock()) as mock_cdll:
                from onnx9000.tensorrt.ffi import TensorRTFFI

                TensorRTFFI()
        sys.platform = old_platform

        # Test error paths
        with patch("ctypes.CDLL", side_effect=OSError("Not found")):
            from onnx9000.tensorrt.ffi import TensorRTFFI

            TensorRTFFI()

    def test_ffi_version_and_plugin(self):
        mock_lib = MagicMock()
        mock_lib.getInferLibVersion.return_value = 8006
        mock_plugin = MagicMock()

        with patch("ctypes.CDLL", side_effect=[mock_lib, mock_plugin]):
            from onnx9000.tensorrt.ffi import TensorRTFFI

            ffi_obj = TensorRTFFI()
            self.assertEqual(ffi_obj.version, (8, 0, 6))

    def test_ops(self):
        print("HELLO WORLD")
        # We test all ops by faking the ffi.lib functions and passing mocked nodes
        net = trt_network.INetworkDefinition(111)
        t1 = trt_network.ITensor(1, "in1")
        t1.ptr = 1
        t2 = trt_network.ITensor(2, "in2")
        t2.ptr = 2
        t3 = trt_network.ITensor(3, "in3")
        t3.ptr = 3
        tensors = {"in1": t1, "in2": t2, "in3": t3}

        class MockNode:
            def __init__(self, op_type, inputs, outputs, attrs=None):
                self.op_type = op_type
                self.inputs = inputs
                self.outputs = outputs
                self.attributes = attrs or {}

        class MockAttr:
            def __init__(self, value):
                self.value = value

            def __iter__(self):
                if isinstance(self.value, list):
                    return iter(self.value)

            def __int__(self):
                return int(self.value)

        # To avoid RuntimeErrors on missing ffi methods, we assign them all
        methods = [
            "addConvolutionNd",
            "addShuffle",
            "addConcatenation",
            "addSlice",
            "addGather",
            "addMatrixMultiply",
            "addElementWise",
            "addUnaryOperation",
            "addActivation",
            "addPoolingNd",
            "addReduce",
        ]
        for m in methods:
            mock_layer = MagicMock()
            mock_layer.getOutput.return_value = 999
            setattr(self.mock_lib, m, MagicMock(return_value=1234))
            setattr(self.mock_lib, "ILayer_getOutput", MagicMock(return_value=999))

        # Also mock getOutput(layer, idx) - we just mock getOutput function globally
        self.mock_lib.ILayer_getOutput = MagicMock(return_value=999)

        # ElementWise Ops
        for op in [
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Max",
            "Min",
            "Pow",
            "Equal",
            "Less",
            "Greater",
            "And",
            "Or",
            "Xor",
        ]:
            node = MockNode(op, ["in1", "in2"], ["out"])
            func = trt.get_op_translator("", op)
            func(net, node, tensors)

        # Missing inputs ElementWise
        node = MockNode("Add", ["in1"], ["out"])
        with self.assertRaises(RuntimeError):
            trt.get_op_translator("", "Add")(net, node, tensors)

        # Unary ops
        for op in [
            "Exp",
            "Log",
            "Sqrt",
            "Abs",
            "Neg",
            "Not",
            "Relu",
            "Sigmoid",
            "Tanh",
            "Elu",
            "Selu",
            "Softplus",
            "HardSigmoid",
        ]:
            node = MockNode(op, ["in1"], ["out"])
            func = trt.get_op_translator("", op)
            if func:
                func(net, node, tensors)

        node = MockNode("LeakyRelu", ["in1"], ["out"], {"alpha": MockAttr(0.1)})
        trt.get_op_translator("", "LeakyRelu")(net, node, tensors)
        node = MockNode("Clip", ["in1"], ["out"], {})
        trt.get_op_translator("", "Clip")(net, node, tensors)

        # Missing inputs unary
        node = MockNode("Relu", [], ["out"])
        with self.assertRaises(RuntimeError):
            trt.get_op_translator("", "Relu")(net, node, tensors)

        # Reduce ops
        for op in ["ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd"]:
            node = MockNode(op, ["in1"], ["out"], {"axes": MockAttr([0]), "keepdims": MockAttr(1)})
            func = trt.get_op_translator("", op)
            if func:
                func(net, node, tensors)

        node = MockNode("ReduceSum", ["in1"], ["out"], {})
        trt.get_op_translator("", "ReduceSum")(net, node, tensors)
        node = MockNode("ReduceSum", ["in1"], ["out"], {"keepdims": MockAttr(0)})
        trt.get_op_translator("", "ReduceSum")(net, node, tensors)

        # Pool ops
        node = MockNode(
            "MaxPool",
            ["in1"],
            ["out"],
            {
                "kernel_shape": MockAttr([2, 2]),
                "strides": MockAttr([2, 2]),
                "pads": MockAttr([0, 0, 0, 0]),
            },
        )
        trt.get_op_translator("", "MaxPool")(net, node, tensors)
        node = MockNode(
            "AveragePool",
            ["in1"],
            ["out"],
            {
                "kernel_shape": MockAttr([2, 2]),
                "strides": MockAttr([2, 2]),
                "pads": MockAttr([0, 0, 0, 0]),
            },
        )
        trt.get_op_translator("", "AveragePool")(net, node, tensors)

        # Gather
        node = MockNode("Gather", ["in1", "in2"], ["out"], {"axis": MockAttr(0)})
        trt.get_op_translator("", "Gather")(net, node, tensors)

        # Slice
        node = MockNode("Slice", ["in1", "in2", "in3"], ["out"], {})
        trt.get_op_translator("", "Slice")(net, node, tensors)

        # Concat
        node = MockNode("Concat", ["in1", "in2"], ["out"], {"axis": MockAttr(0)})
        trt.get_op_translator("", "Concat")(net, node, tensors)

        # Reshape / Transpose
        for op in ["Reshape", "Transpose"]:
            node = MockNode(op, ["in1"], ["out"], {})
            func = trt.get_op_translator("", op)
            if func:
                func(net, node, tensors)

        node = MockNode("Transpose", ["in1"], ["out"], {"perm": MockAttr([1, 0])})
        trt.get_op_translator("", "Transpose")(net, node, tensors)

        # MatMul
        node = MockNode("MatMul", ["in1", "in2"], ["out"], {})
        trt.get_op_translator("", "MatMul")(net, node, tensors)

        # Conv
        node = MockNode(
            "Conv",
            ["in1", "in2", "in3"],
            ["out"],
            {
                "kernel_shape": MockAttr([3, 3]),
                "strides": MockAttr([1, 1]),
                "pads": MockAttr([0, 0, 0, 0]),
                "dilations": MockAttr([1, 1]),
                "group": MockAttr(1),
            },
        )
        trt.get_op_translator("", "Conv")(net, node, tensors)
        node = MockNode("Conv", ["in1", "in2"], ["out"], {"kernel_shape": MockAttr([3, 3])})
        trt.get_op_translator("", "Conv")(net, node, tensors)
        node = MockNode("Conv", ["in1"], ["out"], {})
        with self.assertRaises((RuntimeError, IndexError)):
            trt.get_op_translator("", "Conv")(net, node, tensors)

        # Delete methods to trigger errors for fallback
        for m in methods:
            setattr(self.mock_lib, m, None)

        for op, func in trt.registry._TRT_OP_REGISTRY.items():
            op_name = op[1]
            node = MockNode(op_name, ["in1", "in2", "in3"], ["out"])
            # Remove from mock if it exists
            try:
                func(net, node, tensors)
                pass
            except (RuntimeError, Exception) as e:
                pass
            # Re-add to mock to test missing inputs
            for m in methods:
                setattr(self.mock_lib, m, MagicMock(return_value=1234))
            node2 = MockNode(op_name, [], ["out"])
            try:
                func(net, node2, tensors)
            except Exception:
                pass
            for m in methods:
                setattr(self.mock_lib, m, MagicMock(return_value=0))
            for m in methods:
                setattr(self.mock_lib, m, MagicMock(return_value=0))

    def test_phase(self):
        from onnx9000.tensorrt.ffi import _phase_1_20_bindings

        self.assertTrue(_phase_1_20_bindings())
