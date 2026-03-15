import pytest
from onnx9000.extensions.custom.math_ops import einsum


def test_einsum_empty_list():
    try:
        einsum("i->i", [[]])
    except Exception as e:
        print("exception:", e)
    try:
        einsum("i->i", [[[]]])
    except Exception as e:
        print("exception:", e)


test_einsum_empty_list()
