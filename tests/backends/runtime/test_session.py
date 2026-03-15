import numpy as np
import pytest
import os
import json
from unittest.mock import patch, MagicMock
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.backends.runtime.session import NativeSession, NativeSessionOptions


def create_simple_graph():
    graph = Graph("test")
    graph.inputs = ["X"]
    graph.outputs = ["Y"]
    node = Node("Relu", ["X"], ["Y"], {})
    graph.add_node(node)
    graph.add_tensor(Tensor("X", (2, 2), DType.FLOAT32))
    graph.add_tensor(Tensor("Y", (2, 2), DType.FLOAT32))
    return graph


def test_session_init():
    graph = create_simple_graph()
    opts = NativeSessionOptions()
    opts.device = "cpu"
    session = NativeSession(graph, sess_options=opts)
    assert session.device == "cpu"
    inputs = {"X": np.array([[-1.0, 2.0]], dtype=np.float32)}
    out = session.run(["Y"], inputs)
    np.testing.assert_array_equal(out[0], np.array([[0.0, 2.0]], dtype=np.float32))


def test_session_auto_device():
    graph = create_simple_graph()
    import onnx9000.backends.runtime.session as ses_mod

    with patch.object(ses_mod, "is_cuda_available", return_value=True):
        session = NativeSession(graph)
        assert session.device == "cuda"
    with (
        patch.object(ses_mod, "is_cuda_available", return_value=False),
        patch.object(ses_mod, "is_hip_available", return_value=True),
    ):
        session = NativeSession(graph)
        assert session.device == "rocm"
    with (
        patch.object(ses_mod, "is_cuda_available", return_value=False),
        patch.object(ses_mod, "is_hip_available", return_value=False),
        patch.object(ses_mod, "is_metal_available", return_value=True),
    ):
        session = NativeSession(graph)
        assert session.device == "metal"


def test_session_profiling(tmpdir):
    graph = create_simple_graph()
    opts = NativeSessionOptions()
    opts.device = "cpu"
    opts.enable_profiling = True
    opts.cache_dir = str(tmpdir)
    session = NativeSession(graph, sess_options=opts)
    inputs = {"X": np.array([[-1.0, 2.0]], dtype=np.float32)}
    out = session.run(["Y"], inputs)
    assert len(out) == 1
    out2 = session.run(None, inputs)
    assert len(out2) == 1
    start_time = session.get_profiling_start_time_ns()
    assert isinstance(start_time, int)
    profile_path = session.end_profiling()
    assert os.path.exists(profile_path)
    with open(profile_path, "r") as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["event"] == "run"


def test_session_profiling_disabled():
    graph = create_simple_graph()
    opts = NativeSessionOptions()
    opts.enable_profiling = False
    session = NativeSession(graph, sess_options=opts)
    path = session.end_profiling()
    assert path == ""


def test_session_graph_partition():
    graph = create_simple_graph()
    session = NativeSession(graph)
    g1, g2 = session.partition_graph()
    assert isinstance(g1, Graph)
    assert isinstance(g2, Graph)


def test_session_memory_utilization():
    graph = create_simple_graph()
    session = NativeSession(graph)
    mem = session.profile_memory_utilization()
    assert "arena_bytes" in mem
    assert mem["overhead_bytes"] == 0.0


def test_session_cache_dir_exception():
    opts = NativeSessionOptions()
    opts.device = "cpu"
    opts.cache_dir = "/dev/null/invalid"
    session = NativeSession(Graph("test"), sess_options=opts)
    assert session.device == "cpu"


def test_session_profile_save_exception(tmpdir):
    graph = create_simple_graph()
    opts = NativeSessionOptions()
    opts.device = "cpu"
    opts.enable_profiling = True
    opts.cache_dir = str(tmpdir)
    session = NativeSession(graph, sess_options=opts)
    inputs = {"X": np.array([[-1.0, 2.0]], dtype=np.float32)}
    session.run(None, inputs)
    with patch("json.dump", side_effect=Exception):
        session.end_profiling()


def test_session_auto_device_cpu():
    graph = create_simple_graph()
    import onnx9000.backends.runtime.session as ses_mod

    with (
        patch.object(ses_mod, "is_cuda_available", return_value=False),
        patch.object(ses_mod, "is_hip_available", return_value=False),
        patch.object(ses_mod, "is_metal_available", return_value=False),
        patch.object(ses_mod, "is_accelerate_available", return_value=False),
    ):
        session = NativeSession(graph)
        assert session.device == "cpu"


def test_session_graph_partition_heterogeneous():
    graph = Graph("test")
    node = Node("Add", ["A", "B"], ["C"], {})
    graph.add_node(node)
    session = NativeSession(graph)
    g1, g2 = session.partition_graph()
    assert len(g1.nodes) == 1
    assert len(g2.nodes) == 0
