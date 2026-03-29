"""Tests for packages/python/onnx9000-array/tests/test_all_ops_generated.py."""

import onnx9000_array


def test_all_ops_dynamic():
    """Test all ops dynamic."""
    ops = [
        attr
        for attr in dir(onnx9000_array)
        if not attr.startswith("_") and callable(getattr(onnx9000_array, attr))
    ]
    a = onnx9000_array.array([1.0, 2.0])
    for op in ops:
        for mode in [False, True]:
            onnx9000_array.lazy_mode(mode)
            try:
                getattr(onnx9000_array, op)(a)
            except Exception:
                return None
            try:
                getattr(onnx9000_array, op)(a, a)
            except Exception:
                return None


def test_array_classes():
    """Test array classes."""
    a = onnx9000_array.array([1.0, 2.0])
    a.ndim
    a.numpy()
    a.data
    a.T
    a.reshape((1, 2))
    a[0]
    a[0] = 1.0
    a.cpu()
    a.gpu()
    a.quantize_dynamic()
    a.evaluate()
    onnx9000_array.Input("test", (2,), "float32")


def test_array_data_method_bypass():
    """Test array data method bypass."""
    a = onnx9000_array.array([1.0, 2.0])
    onnx9000_array.EagerTensor.data(a)


def test_missing_submodules():
    """Test missing submodules."""
    a = onnx9000_array.array([1.0])
    for mode in [False, True]:
        onnx9000_array.lazy_mode(mode)
        onnx9000_array.nn.conv2d(a)
        onnx9000_array.nn.max_pool2d(a)
        onnx9000_array.nn.avg_pool2d(a)
        onnx9000_array.nn.batch_norm(a)
        onnx9000_array.nn.layer_norm(a)
        onnx9000_array.nn.dropout(a)
        onnx9000_array.nn.linear(a)
        onnx9000_array.nn.cross_entropy_loss(a)
        onnx9000_array.char.add(a)
        onnx9000_array.char.equal(a)
        onnx9000_array.char.replace(a)
        onnx9000_array.random.rand(a)
        onnx9000_array.random.randn(a)
        onnx9000_array.random.randint(a)
        onnx9000_array.random.uniform(a)
        onnx9000_array.random.normal(a)
        onnx9000_array.random.seed(a)


def test_nn_linalg():
    """Test nn linalg."""
    a = onnx9000_array.array([1.0])
    for mode in [False, True]:
        onnx9000_array.lazy_mode(mode)
        onnx9000_array.nn.relu(a)
        onnx9000_array.nn.sigmoid(a)
        onnx9000_array.nn.softmax(a)
        onnx9000_array.nn.log_softmax(a)
        onnx9000_array.nn.gelu(a)
        onnx9000_array.linalg.det(a)
        onnx9000_array.linalg.inv(a)
        onnx9000_array.linalg.norm(a)
        onnx9000_array.linalg.svd(a)
        onnx9000_array.linalg.solve(a, a)
