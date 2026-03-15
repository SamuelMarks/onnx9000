"""Module providing core logic and structural definitions."""

from onnx9000.backends.web.shaders_tensor import WGSLTensorShaders


def test_reduction_shaders():
    """Provides semantic functionality and verification."""
    for op in ["ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"]:
        s = WGSLTensorShaders.generate_reduction(op, "f32")
        assert "shared_mem" in s
        if op == "ReduceMax":
            assert "-3.402823e+38" in s
    sf16 = WGSLTensorShaders.generate_reduction("ReduceSum", "f16")
    assert "enable f16;" in sf16


def test_manipulation_shaders():
    """Provides semantic functionality and verification."""
    for op in [
        "Reshape",
        "Transpose",
        "Concat",
        "Split",
        "Slice",
        "Gather",
        "ScatterND",
        "Tile",
        "Pad",
        "Cast",
        "Where",
    ]:
        s = WGSLTensorShaders.generate_manipulation(op, "f32")
        assert f"Op: {op}" in s
    s2 = WGSLTensorShaders.generate_manipulation(
        "Reshape", "f32", optimize_aliasing=True
    )
    assert "Optimized memory aliasing" in s2
