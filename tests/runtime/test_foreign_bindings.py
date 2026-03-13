"""Test foreign language bindings parity."""

from pathlib import Path


def test_csharp_bindings():
    """Check that C# stubs contain required classes."""
    p = Path("src/onnx9000/runtime/bindings/csharp/onnxruntime.cs")
    assert p.exists()
    content = p.read_text()
    assert "class InferenceSession" in content
    assert "class SessionOptions" in content
    assert "class RunOptions" in content
    assert "class OrtValue" in content
    assert "class NamedOnnxValue" in content
    assert "IDisposable" in content


def test_java_bindings():
    """Check that Java stubs contain required classes."""
    p = Path("src/onnx9000/runtime/bindings/java/OrtEnvironment.java")
    assert p.exists()
    content = p.read_text()
    assert "class OrtEnvironment" in content
    assert "class OrtSession" in content
    assert "class OnnxTensor" in content
    assert "class OnnxSequence" in content
    assert "class OnnxMap" in content
    assert "class JNIBridges" in content


def test_js_bindings():
    """Check that JS stubs contain required classes."""
    p = Path("src/onnx9000/runtime/bindings/js/onnxruntime-web.js")
    assert p.exists()
    content = p.read_text()
    assert "class InferenceSession" in content
    assert "class Tensor" in content
    assert "env =" in content
    assert "wasm:" in content
    assert "webgl:" in content
    assert "webgpu:" in content
