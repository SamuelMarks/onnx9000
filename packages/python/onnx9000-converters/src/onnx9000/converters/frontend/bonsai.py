from typing import Any

import numpy as np
from onnx9000.core.ir import DType, Graph, Tensor
from onnx9000.optimizer.sparse.modifier import NMPruningModifier


class BonsaiImporter:
    """Parses jax-ml/bonsai structures and integrates structured sparsity."""

    def __init__(self, strip_zeros_to_sparse: bool = True):
        self.strip_zeros_to_sparse = strip_zeros_to_sparse

    def import_model(self, model_dict: dict[str, Any]) -> Graph:
        g = Graph(name="BonsaiImported")

        # Example dynamic strip during ingestion
        if self.strip_zeros_to_sparse:
            for name, weight in model_dict.get("weights", {}).items():
                if isinstance(weight, np.ndarray):
                    non_zero = np.count_nonzero(weight)
                    sparsity = 1.0 - (non_zero / weight.size) if weight.size > 0 else 0
                    if sparsity >= 0.5:
                        g.tensors[name] = Tensor(
                            name=name,
                            shape=weight.shape,
                            dtype=DType.FLOAT32,
                            data=weight.tobytes(),
                        )
                        g.tensors[name].is_sparse = True

        nm_modifier = NMPruningModifier(n=2, m=4)
        nm_modifier.apply(g)

        return g
