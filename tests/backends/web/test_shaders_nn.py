"""Module providing core logic and structural definitions."""

from onnx9000.backends.web.shaders_nn import WGSLNNShaders


def test_matmul_shaders():
    """Provides semantic functionality and verification."""
    f32 = WGSLNNShaders.generate_matmul("f32", 16)
    assert "tileA" in f32
    assert "tileB" in f32
    assert "array<f32, 256>" in f32
    f16 = WGSLNNShaders.generate_matmul("f16", 16)
    assert "enable f16;" in f16
    assert "array<f16, 256>" in f16


def test_conv_shaders():
    """Provides semantic functionality and verification."""
    c = WGSLNNShaders.generate_conv("Conv", "f32", im2col=True)
    assert "mode: Conv, im2col: True, depthwise: False" in c
    assert "tile_cache" in c
    ct = WGSLNNShaders.generate_conv("ConvTranspose", "f16", depthwise=True)
    assert "enable f16;" in ct
    assert "mode: ConvTranspose, im2col: False, depthwise: True" in ct


def test_pool_shaders():
    """Provides semantic functionality and verification."""
    for op in ["MaxPool", "AveragePool", "GlobalAveragePool"]:
        p = WGSLNNShaders.generate_pool(op, "f32")
        assert f"Pool: {op}" in p
        assert "tile_cache" in p


def test_norm_shaders():
    """Provides semantic functionality and verification."""
    for op in ["BatchNorm", "LayerNorm", "InstanceNorm"]:
        n = WGSLNNShaders.generate_norm(op, "f16")
        assert f"Norm: {op}" in n
        assert "enable f16;" in n
        assert "reduction_cache" in n
