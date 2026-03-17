"""Embedding layers."""

from typing import Optional

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Parameter, Tensor
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
        """Implements the __init__ method."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter((num_embeddings, embedding_dim), dtype, "weight")

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        from onnx9000.converters.frontend.utils import record_op

        return record_op("Gather", [self.weight, input], {"axis": 0})
