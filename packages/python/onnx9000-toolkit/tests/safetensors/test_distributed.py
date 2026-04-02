"""Tests for distributed safetensors loading."""

from unittest.mock import patch

from onnx9000.toolkit.safetensors.distributed import load_sharded_tensors, pipeline_parallel_loader


def test_load_sharded_tensors():
    """Verify sharding logic for load_sharded_tensors."""
    # Mock load_file so we don't need a real file
    mock_tensors = {f"tensor_{i}": i for i in range(10)}
    with patch("onnx9000.toolkit.safetensors.distributed.load_file", return_value=mock_tensors):
        # 10 tensors, world size 3 -> sizes: 4, 3, 3
        # rank 0
        r0 = load_sharded_tensors("dummy.safetensors", rank=0, world_size=3)
        assert list(r0.keys()) == ["tensor_0", "tensor_1", "tensor_2", "tensor_3"]
        assert r0["tensor_0"] == 0

        # rank 1
        r1 = load_sharded_tensors("dummy.safetensors", rank=1, world_size=3)
        assert list(r1.keys()) == ["tensor_4", "tensor_5", "tensor_6"]

        # rank 2
        r2 = load_sharded_tensors("dummy.safetensors", rank=2, world_size=3)
        assert list(r2.keys()) == ["tensor_7", "tensor_8", "tensor_9"]


def test_pipeline_parallel_loader():
    """Verify pipeline logic for pipeline_parallel_loader."""
    mock_tensors = {
        "layer_0.weight": 0,
        "layer_0.bias": 1,
        "layer_1.weight": 2,
        "layer_2.weight": 3,
        "other.weight": 4,
    }
    with patch("onnx9000.toolkit.safetensors.distributed.load_file", return_value=mock_tensors):
        r = pipeline_parallel_loader("dummy.safetensors", layers=["layer_1", "layer_2"], rank=1)
        assert "layer_1.weight" in r
        assert "layer_2.weight" in r
        assert "layer_0.weight" not in r
        assert "other.weight" not in r
