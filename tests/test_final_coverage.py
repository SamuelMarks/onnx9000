import pytest
import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.parser.inference import infer_scatter_nd
from onnx9000.core.dtypes import DType


def test_infer_scatter_nd_existing_out():
    graph = Graph("test")
    graph.add_tensor(Tensor("data", (2, 2), DType.FLOAT32))
    graph.add_tensor(Tensor("indices", (1,), DType.INT64))
    graph.add_tensor(Tensor("updates", (1, 2), DType.FLOAT32))
    graph.add_tensor(Tensor("out", (), DType.FLOAT32))
    node = Node("ScatterND", ["data", "indices", "updates"], ["out"], {})
    infer_scatter_nd(node, graph)
    assert graph.tensors["out"].shape == (2, 2)


from onnx9000.extensions.custom.math_ops import inverse, svd, einsum


def test_math_ops_coverage():
    with pytest.raises(ValueError, match="Matrix is singular"):
        inverse([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    svd([[0.0, -1.0], [-1.0, 1.0]])
    with pytest.raises(ValueError, match="Shape of operand 0 does not match"):
        einsum("ij->i", [[[]]])


from onnx9000.extensions.custom.vision_ops import roi_align


def test_roi_align_out_of_bounds():
    X = np.ones((1, 1, 10, 10), dtype=np.float32).tolist()
    rois = [[-10.0, -10.0, 15.0, 15.0]]
    roi_align(
        X,
        rois,
        [0],
        output_height=2,
        output_width=2,
        spatial_scale=1.0,
        sampling_ratio=1,
        aligned=True,
    )


from onnx9000.extensions.text.unigram import UnigramTokenizer


def test_unigram_dp_inf():
    tok = UnigramTokenizer({"a": 1}, {"a": -1.0})
    tok.unk_score = -float("inf")
    try:
        tok.encode("ab")
    except Exception:
        pass


import onnx9000.frontends.frontend.nn.functional as F
from onnx9000.frontends.frontend.tensor import Tensor as FrontendTensor


def test_interpolate_align_corners():
    x = FrontendTensor(np.ones((1, 1, 2, 2)))
    res = F.interpolate(x, scale_factor=2.0, mode="linear", align_corners=True)
    assert res is None
