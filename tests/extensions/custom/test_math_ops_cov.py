import pytest
from onnx9000.extensions.custom.math_ops import einsum


def test_math_ops_missing_lines():
    with pytest.raises(Exception):
        einsum("i->i", [[]])
    with pytest.raises(ValueError, match="Shape of operand 0 does not match term i"):
        einsum("i->i", [[1]])
    with pytest.raises(ValueError, match="Dimension mismatch for index i"):
        einsum("ii->i", [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="Output dimension j not found in inputs"):
        einsum("i->j", [1])
