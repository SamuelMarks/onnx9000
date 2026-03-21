"""Tests the topi functions semantically without dummy asserts."""

import pytest
from onnx9000.tvm.te.tensor import placeholder
from onnx9000.tvm.te.topi import (
    nn_conv2d,
    nn_layer_norm,
    nn_matmul,
    nn_pool2d,
    nn_softmax,
)


def test_nn_conv2d():
    """Verify that nn_conv2d returns a correctly shaped tensor."""
    data = placeholder(shape=(1, 3, 224, 224), dtype="float32", name="data")
    weight = placeholder(shape=(16, 3, 3, 3), dtype="float32", name="weight")

    # Stride 1, Padding 1 all around
    out = nn_conv2d(data, weight, strides=(1, 1), padding=(1, 1, 1, 1))

    # 224 + 2 (padding) - 2 (filter - 1) = 224
    assert out.shape == (1, 16, 224, 224)
    assert out.name == "conv2d"


def test_nn_matmul():
    """Verify that nn_matmul returns a correctly shaped tensor."""
    A = placeholder(shape=(10, 20), dtype="float32", name="A")
    B = placeholder(shape=(20, 30), dtype="float32", name="B")

    out = nn_matmul(A, B)
    assert out.shape == (10, 30)
    assert out.name == "matmul"


def test_nn_pool2d():
    """Verify max and avg pooling output shapes and behavior."""
    data = placeholder(shape=(2, 64, 112, 112), dtype="float32", name="data")

    # Max pooling
    out_max = nn_pool2d(
        data, pool_size=(2, 2), strides=(2, 2), padding=(0, 0, 0, 0), pool_type="max"
    )
    assert out_max.shape == (2, 64, 56, 56)
    assert out_max.name == "max_pool2d"

    # Avg pooling
    out_avg = nn_pool2d(
        data, pool_size=(3, 3), strides=(1, 1), padding=(1, 1, 1, 1), pool_type="avg"
    )
    # 112 + 2 - 3 + 1 = 112
    assert out_avg.shape == (2, 64, 112, 112)
    assert out_avg.name == "avg_pool2d"


def test_nn_softmax():
    """Verify softmax returns a correctly shaped tensor."""
    x = placeholder(shape=(4, 10), dtype="float32", name="x")
    out = nn_softmax(x, axis=-1)

    assert out.shape == (4, 10)
    assert out.name == "softmax"


def test_nn_layer_norm():
    """Verify layer_norm returns a correctly shaped tensor."""
    data = placeholder(shape=(2, 512), dtype="float32", name="data")
    gamma = placeholder(shape=(512,), dtype="float32", name="gamma")
    beta = placeholder(shape=(512,), dtype="float32", name="beta")

    out = nn_layer_norm(data, gamma, beta, axis=-1, epsilon=1e-5)
    assert out.shape == (2, 512)
    assert out.name == "layer_norm"
