"""Embedding layers."""

from typing import Any, Optional
from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.frontends.frontend.tensor import Parameter, Tensor
from onnx9000.core.dtypes import DType


class Embedding(Module):
    """Embedding layer."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = Parameter((num_embeddings, embedding_dim), dtype, "weight")

    def forward(self, input: Tensor) -> Tensor:
        """Provides semantic functionality and verification."""
        from onnx9000.frontends.frontend.utils import record_op

        # Gather (axis=0)
        # weight is (num_embeddings, embedding_dim)
        # input is indices
        # PyTorch Embedding is equivalent to Gather(weight, indices, axis=0)
        return record_op("Gather", [self.weight, input], {"axis": 0})
