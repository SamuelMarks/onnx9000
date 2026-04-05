"""Tests for serializer more."""

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Constant, Graph, Node, SparseTensor, Tensor
from onnx9000.core.serializer import _serialize_shape, save


class TestSerializerMore(unittest.TestCase):
    """Docstring for D101."""

    def test_serialize_shape_float(self):
        """Docstring for D102."""
        shape = _serialize_shape([1.0, -1.0])
        self.assertEqual(shape.dim[0].dim_value, 1)
        self.assertEqual(shape.dim[1].dim_param, "?")

    def test_serialize_sparse_and_external_and_compress(self):
        """Docstring for D102."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            g = Graph("test")

            # Add sparse tensor
            st = SparseTensor("sp")
            st.format = "COO"
            st.shape = (10,)
            st.values = Constant(
                "val",
                values=np.array([1, 2, 3], dtype=np.float32).tobytes(),
                dtype=DType.FLOAT32,
                shape=(3,),
            )
            st.indices = Constant(
                "ind",
                values=np.array([0, 5, 9], dtype=np.int64).tobytes(),
                dtype=DType.INT64,
                shape=(3,),
            )

            g.tensors["sp"] = st
            g.sparse_initializers.append("sp")
            g.sparse_initializers.append("missing_sp")

            # Add large dense tensor
            large_data = b"x" * 2048
            t_large = Constant("large", values=large_data, dtype=DType.UINT8, shape=(2048,))
            g.tensors["large"] = t_large
            g.initializers.append("large")

            # Add attributes TENSOR and GRAPH
            sub_g = Graph("subg")
            t_attr = Tensor("t_attr", [1], DType.FLOAT32)
            n = Node("Op", [], [])
            n.attributes["attr_t"] = Attribute("attr_t", "TENSOR", t_attr)
            n.attributes["attr_g"] = Attribute("attr_g", "GRAPH", sub_g)
            g.add_node(n)

            save(g, path, external_data=True, compress=True)

            zip_path = path.with_suffix(".zip")
            self.assertTrue(zip_path.exists())

            with zipfile.ZipFile(zip_path, "r") as zf:
                self.assertIn("model.onnx", zf.namelist())
                self.assertIn("model.onnx.data", zf.namelist())

            # Run again to hit the `if exists() unlink()` lines
            save(g, path, external_data=True, compress=False)
            self.assertTrue(path.exists())
            self.assertTrue((Path(tmpdir) / "model.onnx.data").exists())
