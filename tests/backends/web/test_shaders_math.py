"""Module providing core logic and structural definitions."""

from onnx9000.backends.web.shaders_math import WGSLMathShaders


def test_unary_shaders():
    """Provides semantic functionality and verification."""
    for op in [
        "Abs",
        "Neg",
        "Sign",
        "Exp",
        "Log",
        "Sqrt",
        "Sin",
        "Cos",
        "Tan",
        "Asin",
        "Acos",
        "Atan",
        "Sinh",
        "Cosh",
        "Asinh",
        "Acosh",
        "Atanh",
        "Erf",
        "IsNaN",
    ]:
        shader_f32 = WGSLMathShaders.generate_unary(op, "f32")
        assert "fn main" in shader_f32
        assert "f32" in shader_f32
        shader_f16 = WGSLMathShaders.generate_unary(op, "f16")
        assert "enable f16;" in shader_f16
        assert "f16" in shader_f16


def test_binary_shaders():
    """Provides semantic functionality and verification."""
    for op in ["Add", "Sub", "Mul", "Div", "Pow", "Mod"]:
        shader_f32 = WGSLMathShaders.generate_binary(op, "f32", broadcast=False)
        assert "fn compute" in shader_f32
        assert "A[index]" in shader_f32
        shader_f16 = WGSLMathShaders.generate_binary(op, "f16", broadcast=True)
        assert "enable f16;" in shader_f16
        assert "broadcast_indices" in shader_f16
