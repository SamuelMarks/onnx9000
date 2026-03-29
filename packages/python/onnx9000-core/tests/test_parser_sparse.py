import unittest

import numpy as np
import onnx9000.core.onnx_pb2 as onnx_pb2
from onnx9000.core.parser.core import _parse_attribute, parse_model, parse_sparse_tensor_proto


class TestParserSparse(unittest.TestCase):
    def test_parse_sparse_tensor(self):
        sparse = onnx_pb2.SparseTensorProto()
        sparse.dims.extend([2, 2])
        sparse.values.data_type = 1
        sparse.values.raw_data = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        sparse.values.name = "val"
        sparse.indices.data_type = 6  # int32
        sparse.indices.raw_data = np.array([0, 3], dtype=np.int32).tobytes()

        parsed = parse_sparse_tensor_proto(sparse)
        self.assertEqual(parsed.shape, (2, 2))

        attr = onnx_pb2.AttributeProto()
        attr.name = "sp"
        attr.type = onnx_pb2.AttributeProto.SPARSE_TENSOR
        attr.sparse_tensor.CopyFrom(sparse)

        pa = _parse_attribute(attr)
        self.assertEqual(pa.attr_type, "SPARSE_TENSOR")

        attr2 = onnx_pb2.AttributeProto()
        attr2.name = "sps"
        attr2.type = onnx_pb2.AttributeProto.SPARSE_TENSORS
        attr2.sparse_tensors.extend([sparse])

        pa2 = _parse_attribute(attr2)
        self.assertEqual(pa2.attr_type, "SPARSE_TENSORS")

    def test_model_sparse(self):
        # We need to test the model with a sparse_initializer
        model = onnx_pb2.ModelProto()
        model.graph.name = "test"

        sparse = onnx_pb2.SparseTensorProto()
        sparse.dims.extend([2, 2])
        sparse.values.data_type = 1
        sparse.values.raw_data = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        sparse.values.name = "val"
        sparse.indices.data_type = 6
        sparse.indices.raw_data = np.array([0, 3], dtype=np.int32).tobytes()
        model.graph.sparse_initializer.extend([sparse])

        ir_model = parse_model(model)
        self.assertEqual(len(ir_model.sparse_initializers), 1)
