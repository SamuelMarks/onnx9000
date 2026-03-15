import numpy as np
import pytest
from onnx9000.backends.cpu.ops import (
    OP_REGISTRY,
    im2col,
    add_op,
    sub_op,
    mul_op,
    div_op,
    pow_op,
    matmul_op,
    conv_op,
    relu_op,
    sigmoid_op,
    tanh_op,
    gelu_op,
    reducesum_op,
    reducemean_op,
    reducemax_op,
    transpose_op,
    reshape_op,
    flatten_op,
    concat_op,
    gather_op,
    scatternd_op,
    slice_op,
    softmax_op,
    layernorm_op,
    batchnorm_op,
)


def test_add():
    res = OP_REGISTRY["Add"]([np.array([1, 2]), np.array([3, 4])], {})
    np.testing.assert_array_equal(res[0], np.array([4, 6]))


def test_sub():
    res = OP_REGISTRY["Sub"]([np.array([1, 2]), np.array([3, 4])], {})
    np.testing.assert_array_equal(res[0], np.array([-2, -2]))


def test_mul():
    res = OP_REGISTRY["Mul"]([np.array([1, 2]), np.array([3, 4])], {})
    np.testing.assert_array_equal(res[0], np.array([3, 8]))


def test_div():
    res = OP_REGISTRY["Div"]([np.array([1, 2]), np.array([2, 4])], {})
    np.testing.assert_array_equal(res[0], np.array([0.5, 0.5]))


def test_pow():
    res = OP_REGISTRY["Pow"]([np.array([2, 3]), np.array([3, 2])], {})
    np.testing.assert_array_equal(res[0], np.array([8, 9]))


def test_matmul():
    res = OP_REGISTRY["MatMul"](
        [np.array([[1, 2], [3, 4]]), np.array([[2, 0], [1, 2]])], {}
    )
    np.testing.assert_array_equal(res[0], np.array([[4, 4], [10, 8]]))


def test_im2col():
    x = np.arange(16).reshape(1, 1, 4, 4)
    res = im2col(x, 2, 2, 1, 1, 0, 0)
    assert res.shape == (9, 4)


def test_conv():
    x = np.ones((1, 1, 3, 3))
    w = np.ones((1, 1, 2, 2))
    b = np.array([1])
    res = conv_op([x, w, b], {"strides": [1, 1], "pads": [0, 0, 0, 0], "group": 1})
    assert res[0].shape == (1, 1, 2, 2)
    np.testing.assert_array_equal(res[0], np.full((1, 1, 2, 2), 5.0))


def test_conv_grouped():
    x = np.ones((1, 2, 3, 3))
    w = np.ones((2, 1, 2, 2))
    res = conv_op([x, w], {"strides": [1, 1], "pads": [0, 0, 0, 0], "group": 2})
    assert res[0].shape == (1, 2, 2, 2)
    np.testing.assert_array_equal(res[0], np.full((1, 2, 2, 2), 4.0))


def test_relu():
    res = OP_REGISTRY["Relu"]([np.array([-1, 2, -3])], {})
    np.testing.assert_array_equal(res[0], np.array([0, 2, 0]))


def test_sigmoid():
    res = OP_REGISTRY["Sigmoid"]([np.array([0])], {})
    np.testing.assert_array_equal(res[0], np.array([0.5]))


def test_tanh():
    res = OP_REGISTRY["Tanh"]([np.array([0])], {})
    np.testing.assert_array_equal(res[0], np.array([0]))


def test_gelu():
    x = np.array([-1.0, 0.0, 1.0])
    res = gelu_op([x], {})
    assert len(res) == 1
    np.testing.assert_allclose(res[0], [-0.158655, 0.0, 0.841345], rtol=0.001)


def test_reducesum():
    x = np.array([[1, 2], [3, 4]])
    res = OP_REGISTRY["ReduceSum"]([x], {"axes": [0], "keepdims": 0})
    np.testing.assert_array_equal(res[0], np.array([4, 6]))
    res2 = OP_REGISTRY["ReduceSum"]([x], {})
    np.testing.assert_array_equal(res2[0], np.array([[10]]))


def test_reducemax():
    x = np.array([[1, 2], [3, 4]])
    res = reducemax_op([x], {"axes": [0], "keepdims": 0})
    np.testing.assert_array_equal(res[0], np.array([3, 4]))


def test_reducemean():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = reducemean_op([x], {"axes": [1], "keepdims": 1})
    np.testing.assert_array_equal(res[0], np.array([[1.5], [3.5]]))


def test_transpose():
    x = np.array([[1, 2], [3, 4]])
    res = OP_REGISTRY["Transpose"]([x], {"perm": [1, 0]})
    np.testing.assert_array_equal(res[0], np.array([[1, 3], [2, 4]]))


def test_reshape():
    x = np.array([[1, 2], [3, 4]])
    shape = np.array([4])
    res = OP_REGISTRY["Reshape"]([x, shape], {})
    np.testing.assert_array_equal(res[0], np.array([1, 2, 3, 4]))
    shape2 = np.array([2, 0])
    res2 = OP_REGISTRY["Reshape"]([x, shape2], {})
    np.testing.assert_array_equal(res2[0], np.array([[1, 2], [3, 4]]))


def test_flatten():
    x = np.array([[[1, 2], [3, 4]]])
    res = flatten_op([x], {"axis": 1})
    np.testing.assert_array_equal(res[0], np.array([[1, 2, 3, 4]]))
    res2 = flatten_op([x], {"axis": -1})
    np.testing.assert_array_equal(res2[0], np.array([[1, 2], [3, 4]]))


def test_concat():
    x = np.array([[1, 2]])
    y = np.array([[3, 4]])
    res = OP_REGISTRY["Concat"]([x, y], {"axis": 0})
    np.testing.assert_array_equal(res[0], np.array([[1, 2], [3, 4]]))


def test_gather():
    x = np.array([[1, 2], [3, 4]])
    indices = np.array([1, 0])
    res = OP_REGISTRY["Gather"]([x, indices], {"axis": 0})
    np.testing.assert_array_equal(res[0], np.array([[3, 4], [1, 2]]))


def test_scatternd():
    data = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 2])
    res = scatternd_op([data, indices, updates], {})
    expected = np.array([[1, 0], [0, 2]])
    np.testing.assert_array_equal(res[0], expected)


def test_slice():
    x = np.arange(10)
    starts = np.array([2])
    ends = np.array([8])
    axes = np.array([0])
    steps = np.array([2])
    res = slice_op([x, starts, ends, axes, steps], {})
    np.testing.assert_array_equal(res[0], np.array([2, 4, 6]))
    res2 = slice_op([x, starts, ends], {})
    np.testing.assert_array_equal(res2[0], np.array([2, 3, 4, 5, 6, 7]))


def test_softmax():
    x = np.array([[1.0, 2.0]])
    res = OP_REGISTRY["Softmax"]([x], {"axis": 1})
    np.testing.assert_allclose(res[0], np.array([[0.26894142, 0.73105858]]))


def test_layernorm():
    x = np.array([[1.0, 3.0]])
    scale = np.array([1.0, 1.0])
    b = np.array([0.0, 0.0])
    res = layernorm_op([x, scale, b], {"axis": -1, "epsilon": 0.0})
    np.testing.assert_allclose(res[0], np.array([[-1.0, 1.0]]))
    res2 = layernorm_op([x, scale], {"axis": -1, "epsilon": 0.0})
    np.testing.assert_allclose(res2[0], np.array([[-1.0, 1.0]]))


def test_batchnorm():
    x = np.array([[[[1.0]], [[3.0]]]])
    scale = np.array([1.0, 1.0])
    b = np.array([0.0, 0.0])
    mean = np.array([1.0, 3.0])
    var = np.array([1.0, 1.0])
    res = batchnorm_op([x, scale, b, mean, var], {"epsilon": 0.0})
    np.testing.assert_allclose(res[0], np.array([[[[0.0]], [[0.0]]]]))


def test_reducemean_no_axes():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = reducemean_op([x], {})
    np.testing.assert_array_equal(res[0], np.array([[2.5]]))


def test_reducemax_no_axes():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = reducemax_op([x], {})
    np.testing.assert_array_equal(res[0], np.array([[4.0]]))
