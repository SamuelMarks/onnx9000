"""Module providing core logic and structural definitions."""

import pytest
import asyncio
from onnx9000.backends.web.api import (
    Tensor,
    Env,
    SessionOptions,
    InferenceSession,
    Tooling,
)


def test_tensor():
    """Provides semantic functionality and verification."""
    t = Tensor("float32", [1.0, 2.0], [2])
    assert t.type == "float32"
    assert t.data == [1.0, 2.0]
    assert t.dims == [2]


def test_env_singleton():
    """Provides semantic functionality and verification."""
    e1 = Env()
    e2 = Env()
    assert e1 is e2
    assert e1.logLevel == "warning"


def test_session_options():
    """Provides semantic functionality and verification."""
    opts = SessionOptions()
    assert "webgpu" in opts.executionProviders
    assert opts.intraOpNumThreads == 1
    assert opts.graphOptimizationLevel == "ORT_ENABLE_ALL"


def test_inference_session():
    """Provides semantic functionality and verification."""

    async def run():
        """Provides semantic functionality and verification."""
        opts = SessionOptions()
        sess = await InferenceSession.create("model.onnx", opts)
        assert sess.model_path == "model.onnx"
        t = Tensor("float32", [1.0], [1])
        res = await sess.run({"input": t})
        assert "input" in res
        assert res["input"].data == [1.0]
        sess2 = await InferenceSession.create("model2.onnx")
        assert sess2.options.intraOpNumThreads == 1

    asyncio.run(run())


def test_tooling():
    """Provides semantic functionality and verification."""
    assert Tooling.dump_profiler() == "{}"
