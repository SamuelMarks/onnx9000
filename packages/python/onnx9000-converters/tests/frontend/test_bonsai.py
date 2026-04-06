import numpy as np
import pytest
from onnx9000.converters.frontend.bonsai import BonsaiImporter


def test_bonsai_importer():
    # 2:4 sparsity pattern mock
    w = np.array([[1, 0, 0, 2], [0, 3, 4, 0]], dtype=np.float32)

    importer = BonsaiImporter()
    graph = importer.import_model({"weights": {"w1": w}})

    assert "w1" in graph.tensors
    assert getattr(graph.tensors["w1"], "is_sparse", False) is True
