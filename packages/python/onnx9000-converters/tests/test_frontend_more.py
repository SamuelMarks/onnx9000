"""Tests for frontend more."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx9000.converters.frontend import builder, exporter, jit, models, tensor, tracer, utils
from onnx9000.converters.frontend.nn import (
    containers,
    conv,
    dropout,
    embedding,
    flatten,
    functional,
    identity,
    init,
    linear,
    module,
    normalization,
    pool,
    rnn,
)


class TestFrontendMore(unittest.TestCase):
    """Represents test frontend more inheriting from unittest.TestCase, supporting operations such as test_builder, test_exporter, test_jit, test_models."""

    def test_builder(self) -> None:
        """Verify that builder behaves as expected."""
        gb = builder.GraphBuilder("test_graph")
        t1 = tensor.Tensor((1, 2), name="t1")
        t2 = tensor.Tensor((2, 1), name="t2")
        gb.inputs.append(t1)
        gb.outputs.append(t2)
        self.assertEqual(gb.name, "test_graph")

        n = tensor.Node("Relu", [t1], [t2], {})
        gb.add_node(n)
        graph = gb.to_graph()
        self.assertEqual(graph.name, "test_graph")

    def test_exporter(self) -> None:
        """Verify that exporter behaves as expected.

        Returns:
            Result of test exporter.
        """

        class SimpleModule(module.Module):
            """Simple module."""

            def __init__(self) -> None:
                """Initialize SimpleModule with configured parameters and state."""
                super().__init__()
                self.l = linear.Linear(2, 2)

            def forward(
                self, x: int | float | str | bool | bytes | list | dict | tuple | None
            ) -> int | float | str | bool | bytes | list | dict | tuple | None:
                """Forward."""
                return self.l(x)

        m = SimpleModule()
        t = tensor.Tensor((1, 2))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.onnx"
            # Trace the model first using exporter logic
            exporter.export(
                m,
                t,
                p,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}},
                custom_opsets={"custom": 1},
            )
            self.assertTrue(p.exists())

    def test_jit(self) -> None:
        """Verify that jit behaves as expected.

        Returns:
            Result of test jit.
        """

        @jit.jit
        def my_func(
            x: int | float | str | bool | bytes | list | dict | tuple | None,
        ) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """My func."""
            return functional.relu(x)

        t = tensor.Tensor((1, 2))
        res = my_func(t)
        self.assertIsInstance(res, builder.GraphBuilder)

    def test_models(self) -> None:
        """Verify that models behaves as expected."""
        # resnet contains conv, pool, linear, etc.
        resnet = models.ResNet18()
        self.assertIsInstance(resnet, module.Module)

    def test_nn_containers_identity_flatten(self) -> None:
        """Verify that nn containers identity flatten behaves as expected."""
        s = containers.Sequential(linear.Linear(2, 2), identity.Identity(), flatten.Flatten())
        t = tensor.Tensor((1, 2))
        # Need to trace it to get output
        with builder.Tracing():
            out = s(t)
            self.assertIsInstance(out, tensor.Tensor)

    def test_nn_conv_pool_normalization(self) -> None:
        """Verify that nn conv pool normalization behaves as expected."""
        c = conv.Conv2d(3, 16, 3)
        mp = pool.MaxPool2d(2)
        bn = normalization.BatchNorm2d(16)

        t = tensor.Tensor((1, 3, 32, 32))
        with builder.Tracing():
            x = c(t)
            x = bn(x)
            x = mp(x)
            self.assertIsInstance(x, tensor.Tensor)

    def test_nn_dropout_embedding_rnn(self) -> None:
        """Verify that nn dropout embedding rnn behaves as expected."""
        d = dropout.Dropout(0.5)
        e = embedding.Embedding(10, 3)
        rn = rnn.RNN(10, 20)

        t = tensor.Tensor((1, 10))
        with builder.Tracing():
            d(t)
            # Try to test e and rn if possible, or just instantiate them
            self.assertIsInstance(e, embedding.Embedding)
            self.assertIsInstance(rn, rnn.RNN)

    def test_init(self) -> None:
        """Verify that init behaves as expected."""
        t = tensor.Tensor((2, 2))
        # Since t.data is None initially, we can just check it doesn't crash,
        # or mock data. Let's provide some numpy data.
        t.data = np.zeros((2, 2), dtype=np.float32)
        init.kaiming_normal_(t)
        init.zeros_(t)
        self.assertTrue((t.data == 0).all())

    def test_functional(self) -> None:
        """Verify that functional behaves as expected."""
        t = tensor.Tensor((1, 2))
        with builder.Tracing():
            res = functional.relu(t)
            self.assertIsInstance(res, tensor.Tensor)

    def test_tracer(self) -> None:
        """Verify that tracer behaves as expected.

        Returns:
            Result of test tracer.
        """

        def simple_func(
            x: int | float | str | bool | bytes | list | dict | tuple | None,
        ) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Simple func."""
            return functional.relu(x)

        t = tensor.Tensor((1, 2))
        gb = tracer.trace(simple_func, t)
        self.assertTrue(len(gb.nodes) > 0)

    def test_utils(self) -> None:
        """Verify that utils behaves as expected."""
        res = utils.infer_elementwise_shape((1, 3), (2, 1))
        self.assertEqual(res, (2, 3))
        res_matmul = utils.infer_matmul_shape((2, 3), (3, 4))
        self.assertEqual(res_matmul, (2, 4))
