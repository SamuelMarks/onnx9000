import unittest
import onnx9000.tensorrt as trt


class TestTensorRTFFI(unittest.TestCase):
    def test_ffi_initialization(self):
        self.assertIsNotNone(trt.ffi)
        self.assertIsInstance(trt.ffi.pointers, dict)

    def test_enums(self):
        self.assertEqual(trt.DataType.kFLOAT.value, 0)
        self.assertEqual(trt.ElementWiseOperation.kSUM.value, 0)

    def test_structs(self):
        dims = trt.Dims([1, 2, 3])
        self.assertEqual(dims.nbDims, 3)
        self.assertEqual(dims.d[0], 1)
        self.assertEqual(dims.d[1], 2)
        self.assertEqual(dims.d[2], 3)

    def test_registry(self):
        def dummy_op(network, node, tensors):
            return True

        trt.register_op("", "Dummy")(dummy_op)
        func = trt.get_op_translator("", "Dummy")
        self.assertIsNotNone(func)


if __name__ == "__main__":
    unittest.main()
