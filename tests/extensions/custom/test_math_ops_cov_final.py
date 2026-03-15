import pytest
from onnx9000.extensions.custom.math_ops import einsum


def test_math_ops_cov_final():
    assert einsum("ij->ij", [[]]) == [[]]
    with pytest.raises(ValueError, match="Dimension mismatch for index i"):
        einsum("i,i->i", [1], [1, 2])
    with pytest.raises(ValueError, match="Output dimension j not found in inputs"):
        einsum("i->j", [1])
