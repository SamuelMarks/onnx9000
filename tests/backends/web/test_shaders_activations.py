"""Module providing core logic and structural definitions."""

from onnx9000.backends.web.shaders_activations import WGSLActivationShaders


def test_activations_shaders():
    """Provides semantic functionality and verification."""
    ops = [
        "Relu",
        "Sigmoid",
        "Tanh",
        "LeakyRelu",
        "Elu",
        "Selu",
        "Softplus",
        "HardSigmoid",
        "Gelu",
    ]
    for op in ops:
        shader_f32 = WGSLActivationShaders.generate(op, "f32", in_place=False)
        assert "fn compute" in shader_f32
        assert "f32" in shader_f32
        assert "var<storage, read> A" in shader_f32
        assert "var<storage, read_write> Y" in shader_f32
        shader_f16 = WGSLActivationShaders.generate(op, "f16", in_place=True)
        assert "enable f16;" in shader_f16
        assert "var<storage, read_write> A" in shader_f16
        assert "A[index] = compute(A[index])" in shader_f16


def test_softmax_shaders():
    """Provides semantic functionality and verification."""
    for op in ["Softmax", "LogSoftmax"]:
        shader_f32 = WGSLActivationShaders.generate(op, "f32", in_place=False)
        assert "@compute" in shader_f32
        if op == "LogSoftmax":
            assert "log(val)" in shader_f32
        shader_f16 = WGSLActivationShaders.generate(op, "f16", in_place=True)
        assert "enable f16;" in shader_f16
        assert "var<storage, read_write> A" in shader_f16
