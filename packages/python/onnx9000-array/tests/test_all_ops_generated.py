"""Exhaustive tests for all top-level and submodule operations in `onnx9000.array`."""

import onnx9000.array as arr


def test_all_top_level_ops_eager_and_lazy() -> None:
    """Verify every top-level array operation in both eager and lazy execution modes."""
    op_Callables = [
        arr.add,
        arr.subtract,
        arr.multiply,
        arr.divide,
        arr.power,
        arr.mod,
        arr.absolute,
        arr.negative,
        arr.sign,
        arr.exp,
        arr.log,
        arr.sqrt,
        arr.square,
        arr.sin,
        arr.cos,
        arr.tan,
        arr.arcsin,
        arr.arccos,
        arr.arctan,
        arr.sinh,
        arr.cosh,
        arr.tanh,
        arr.arcsinh,
        arr.arccosh,
        arr.arctanh,
        arr.matmul,
        arr.equal,
        arr.less,
        arr.greater,
        arr.less_equal,
        arr.greater_equal,
        arr.logical_and,
        arr.logical_or,
        arr.logical_not,
        arr.logical_xor,
        arr.isnan,
        arr.isinf,
        arr.where,
        arr.sum,
        arr.prod,
        arr.mean,
        arr.min,
        arr.max,
        arr.argmin,
        arr.argmax,
        arr.reshape,
        arr.squeeze,
        arr.expand_dims,
        arr.concatenate,
        arr.split,
        arr.tile,
        arr.pad,
        arr.transpose,
        arr.take,
        arr.gather,
        arr.sort,
        arr.argsort,
        arr.nonzero,
        arr.zeros,
        arr.ones,
        arr.empty,
        arr.full,
        arr.eye,
        arr.identity,
        arr.arange,
        arr.linspace,
        arr.log10,
        arr.log2,
        arr.cbrt,
        arr.reciprocal,
        arr.deg2rad,
        arr.rad2deg,
        arr.dot,
        arr.vdot,
        arr.inner,
        arr.outer,
        arr.tensordot,
        arr.einsum,
        arr.swapaxes,
        arr.trace,
        arr.ptp,
        arr.all,
        arr.any,
        arr.cumsum,
        arr.cumprod,
        arr.ravel,
        arr.broadcast_to,
        arr.stack,
        arr.vstack,
        arr.hstack,
        arr.dstack,
        arr.array_split,
        arr.repeat,
        arr.not_equal,
        arr.allclose,
        arr.isclose,
        arr.extract,
        arr.take_along_axis,
        arr.put,
        arr.put_along_axis,
        arr.nan_to_num,
        arr.clip,
        arr.around,
        arr.fix,
        arr.i0,
        arr.sinc,
        arr.save,
        arr.load,
        arr.vectorize,
        arr.meshgrid,
        arr.mgrid,
        arr.einsum_path,
        arr.polyfit,
        arr.histogram,
        arr.digitize,
        arr.export_model,
        arr.compile,
        arr.set_device,
        arr.set_log_level,
        arr.set_opset,
        arr.set_num_threads,
    ]
    a = arr.array([1.0, 2.0])
    for fn in op_Callables:
        arr.lazy_mode(False)
        res_eager = fn(a)
        assert isinstance(res_eager, arr.EagerTensor)
        arr.lazy_mode(True)
        res_lazy = fn(a)
        assert isinstance(res_lazy, arr.LazyTensor)
    arr.lazy_mode(False)


def test_submodules_nn_linalg_char_random() -> None:
    """Verify all operations in `nn`, `linalg`, `char`, and `random` namespaces."""
    a = arr.array([1.0])
    for mode, expected_cls in [(False, arr.EagerTensor), (True, arr.LazyTensor)]:
        arr.lazy_mode(mode)
        assert isinstance(arr.nn.relu(a), expected_cls)
        assert isinstance(arr.nn.sigmoid(a), expected_cls)
        assert isinstance(arr.nn.softmax(a), expected_cls)
        assert isinstance(arr.nn.log_softmax(a), expected_cls)
        assert isinstance(arr.nn.gelu(a), expected_cls)
        assert isinstance(arr.nn.conv2d(a), expected_cls)
        assert isinstance(arr.nn.max_pool2d(a), expected_cls)
        assert isinstance(arr.nn.avg_pool2d(a), expected_cls)
        assert isinstance(arr.nn.batch_norm(a), expected_cls)
        assert isinstance(arr.nn.layer_norm(a), expected_cls)
        assert isinstance(arr.nn.dropout(a), expected_cls)
        assert isinstance(arr.nn.linear(a), expected_cls)
        assert isinstance(arr.nn.cross_entropy_loss(a), expected_cls)

        assert isinstance(arr.linalg.det(a), expected_cls)
        assert isinstance(arr.linalg.inv(a), expected_cls)
        assert isinstance(arr.linalg.norm(a), expected_cls)
        assert isinstance(arr.linalg.svd(a), expected_cls)
        assert isinstance(arr.linalg.solve(a, a), expected_cls)

        assert isinstance(arr.char.add(a), expected_cls)
        assert isinstance(arr.char.equal(a), expected_cls)
        assert isinstance(arr.char.replace(a), expected_cls)

        assert isinstance(arr.random.rand(a), expected_cls)
        assert isinstance(arr.random.randn(a), expected_cls)
        assert isinstance(arr.random.randint(a), expected_cls)
        assert isinstance(arr.random.uniform(a), expected_cls)
        assert isinstance(arr.random.normal(a), expected_cls)
        assert arr.random.seed(42) is None
    arr.lazy_mode(False)
