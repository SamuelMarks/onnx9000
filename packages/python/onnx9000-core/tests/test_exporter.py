"""Tests for the Unified Exporter."""

import os
from pathlib import Path

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.exporter import export_graph, generate_keras, generate_mlir
from onnx9000.core.ir import Graph, Node, Tensor, Variable


def test_export_onnx(tmp_path):
    """Test exporting to ONNX format."""
    g = Graph("test")
    v_in = Variable("in", shape=[1, 3], dtype=DType.FLOAT32)
    v_out = Variable("out", shape=[1, 3], dtype=DType.FLOAT32)
    g.add_tensor(v_in)
    g.add_tensor(v_out)
    g.inputs.append("in")
    g.outputs.append("out")

    n = Node("Relu", inputs=["in"], outputs=["out"])
    g.add_node(n)

    out_path = str(tmp_path / "model.onnx")
    export_graph(g, out_path, "onnx")
    assert os.path.exists(out_path)


def test_export_c(tmp_path):
    """Test exporting to C format."""
    g = Graph("test")
    v_in = Variable("in", shape=[1, 3], dtype=DType.FLOAT32)
    v_out = Variable("out", shape=[1, 3], dtype=DType.FLOAT32)
    g.add_tensor(v_in)
    g.add_tensor(v_out)
    g.inputs.append("in")
    g.outputs.append("out")
    n = Node("Relu", inputs=["in"], outputs=["out"])
    g.add_node(n)

    out_path = str(tmp_path / "model.c")
    export_graph(g, out_path, "c")
    assert os.path.exists(out_path)
    assert os.path.exists(out_path.replace(".c", ".h"))


def test_export_cpp(tmp_path):
    """Test exporting to C++ format."""
    g = Graph("test")
    v_in = Variable("in", shape=[1, 3], dtype=DType.FLOAT32)
    v_out = Variable("out", shape=[1, 3], dtype=DType.FLOAT32)
    g.add_tensor(v_in)
    g.add_tensor(v_out)
    g.inputs.append("in")
    g.outputs.append("out")
    n = Node("Relu", inputs=["in"], outputs=["out"])
    g.add_node(n)

    out_path = str(tmp_path / "model.cpp")
    export_graph(g, out_path, "cpp")
    assert os.path.exists(out_path)


def test_export_wasm(tmp_path, monkeypatch):
    """Test exporting to WASM format."""
    g = Graph("test")
    v_in = Variable("in", shape=[1, 3], dtype=DType.FLOAT32)
    v_out = Variable("out", shape=[1, 3], dtype=DType.FLOAT32)
    g.add_tensor(v_in)
    g.add_tensor(v_out)
    g.inputs.append("in")
    g.outputs.append("out")
    n = Node("Relu", inputs=["in"], outputs=["out"])
    g.add_node(n)

    out_path = str(tmp_path / "model.js")

    # Mock compile_wasm
    def mock_compile_wasm(graph, out_dir):
        p = Path(out_dir) / "temp.js"
        p.write_text("// dummy js")
        (Path(out_dir) / "temp.wasm").write_bytes(b"dummy wasm")
        return p

    import onnx9000.converters.jit.compiler as compiler

    monkeypatch.setattr(compiler, "compile_wasm", mock_compile_wasm)

    export_graph(g, out_path, "wasm")
    assert os.path.exists(out_path)
    assert os.path.exists(out_path.replace(".js", ".wasm"))


def test_export_keras(tmp_path):
    """Test exporting to Keras format."""
    g = Graph("test")
    v_in = Variable("in", shape=[1, 32, 224, 224], dtype=DType.FLOAT32)
    v_out = Variable("out", shape=[1, 10], dtype=DType.FLOAT32)
    g.add_tensor(v_in)
    g.add_tensor(v_out)
    g.inputs.append("in")
    g.outputs.append("out")

    # Test multiple op types for coverage
    n1 = Node("Conv", inputs=["in"], outputs=["c_out"])
    n2 = Node("Relu", inputs=["c_out"], outputs=["r_out"])
    n3 = Node("Add", inputs=["r_out", "r_out"], outputs=["a_out"])
    n4 = Node("MatMul", inputs=["a_out"], outputs=["out"])
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)

    out_path = str(tmp_path / "model.py")
    export_graph(g, out_path, "keras")
    assert os.path.exists(out_path)
    content = Path(out_path).read_text()
    assert "def get_model_test():" in content
    assert "Conv2D" in content
    assert "ops.relu" in content
    assert "Dense" in content


def test_export_mlir(tmp_path):
    """Test exporting to MLIR format."""
    g = Graph("test")
    v_in = Variable("in", shape=[1, 3], dtype=DType.FLOAT32)
    v_out = Variable("out", shape=[1, 3], dtype=DType.FLOAT32)
    g.add_tensor(v_in)
    g.add_tensor(v_out)
    g.inputs.append("in")
    g.outputs.append("out")
    n = Node("Relu", inputs=["in"], outputs=["out"])
    g.add_node(n)

    out_path = str(tmp_path / "model.mlir")
    export_graph(g, out_path, "mlir")
    assert os.path.exists(out_path)
    content = Path(out_path).read_text()
    assert "module {" in content
    assert "onnx.Relu" in content


def test_export_unsupported():
    """Test exporting to an unsupported format."""
    g = Graph("test")
    with pytest.raises(ValueError, match="Unsupported format"):
        export_graph(g, "out", "invalid")


def test_export_c_no_ext(tmp_path):
    """Test exporting to C format with no .c extension to trigger header_path += '.h'."""
    g = Graph("test")
    out_path = str(tmp_path / "model")
    export_graph(g, out_path, "c")
    assert os.path.exists(out_path)
    assert os.path.exists(out_path + ".h")


def test_export_wasm_existing(tmp_path, monkeypatch):
    """Test exporting to WASM format with existing files to trigger os.remove."""
    g = Graph("test")
    out_path = str(tmp_path / "model.js")
    wasm_path = str(tmp_path / "model.wasm")

    Path(out_path).write_text("old js")
    Path(wasm_path).write_bytes(b"old wasm")

    # Mock compile_wasm to return a different path
    def mock_compile_wasm(graph, out_dir):
        p = Path(out_dir) / "temp.js"
        p.write_text("// dummy js")
        (Path(out_dir) / "temp.wasm").write_bytes(b"dummy wasm")
        return p

    import onnx9000.converters.jit.compiler as compiler

    monkeypatch.setattr(compiler, "compile_wasm", mock_compile_wasm)

    export_graph(g, out_path, "wasm")
    assert Path(out_path).read_text() == "// dummy js"
    assert Path(wasm_path).read_bytes() == b"dummy wasm"


def test_generate_keras_multiple_inputs():
    """Test generate_keras with multiple inputs."""
    g = Graph("multi")
    g.inputs = ["in1", "in2"]
    g.tensors["in1"] = Variable("in1", shape=[1, 10], dtype=DType.FLOAT32)
    g.tensors["in2"] = Variable("in2", shape=[1, 10], dtype=DType.FLOAT32)
    code = generate_keras(g)
    assert "model = keras.Model(inputs=[input_0, input_1]" in code


def test_generate_keras_no_inputs():
    """Test generate_keras with no inputs."""
    g = Graph("none")
    code = generate_keras(g)
    assert "model = keras.Model(inputs=[], outputs=[])" in code


def test_generate_keras_unknown_op():
    """Test generate_keras with an unknown operator."""
    g = Graph("unknown")
    v_in = Variable("in", shape=[1, 3], dtype=DType.FLOAT32)
    v_out = Variable("out", shape=[1, 3], dtype=DType.FLOAT32)
    g.add_tensor(v_in)
    g.add_tensor(v_out)
    g.inputs.append("in")
    g.outputs.append("out")
    g.add_node(Node("UnknownOp", inputs=["in"], outputs=["out"]))
    code = generate_keras(g)
    assert "Fallback for UnknownOp" in code


def test_export_c_format(tmp_path):
    """Docstring for D103."""
    g = Graph("test")
    x = Variable("x", shape=(1,), dtype=DType.FLOAT32)
    y = Variable("y", shape=(1,), dtype=DType.FLOAT32)
    z = Variable("z", shape=(1,), dtype=DType.FLOAT32)
    g.add_tensor(x)
    g.add_tensor(y)
    g.add_tensor(z)
    g.nodes.append(Node("Add", inputs=["x", "y"], outputs=["z"]))
    out_path = str(tmp_path / "model.c")
    export_graph(g, out_path, "c")
    assert os.path.exists(out_path)


def test_export_c_format_same_name(tmp_path):
    """Docstring for D103."""
    g = Graph("test")
    x = Variable("x", shape=(1,), dtype=DType.FLOAT32)
    y = Variable("y", shape=(1,), dtype=DType.FLOAT32)
    z = Variable("z", shape=(1,), dtype=DType.FLOAT32)
    g.add_tensor(x)
    g.add_tensor(y)
    g.add_tensor(z)
    g.nodes.append(Node("Add", inputs=["x", "y"], outputs=["z"]))
    out_path = str(tmp_path / "model")
    export_graph(g, out_path, "c")
    assert os.path.exists(out_path)


def test_export_wasm_format(tmp_path):
    """Docstring for D103."""
    g = Graph("test")
    x = Variable("x", shape=(1,), dtype=DType.FLOAT32)
    y = Variable("y", shape=(1,), dtype=DType.FLOAT32)
    z = Variable("z", shape=(1,), dtype=DType.FLOAT32)
    g.add_tensor(x)
    g.add_tensor(y)
    g.add_tensor(z)
    g.nodes.append(Node("Add", inputs=["x", "y"], outputs=["z"]))
    out_path = str(tmp_path / "model.wasm")

    # We will need to mock subprocess for this otherwise it will fail expecting emcc
    import shutil
    import subprocess
    import unittest.mock

    def fake_which(cmd):
        if cmd == "emcc":
            return "/usr/bin/emcc"
        return getattr(shutil, "_which", lambda x: "/usr/bin/emcc")(cmd)

    def fake_run(*args, **kwargs):
        import os

        from onnx9000.converters.jit.compiler import hash_graph

        cache_key = hash_graph(g)
        js_path = os.path.join(os.path.dirname(out_path), f"onnx9000_{cache_key}.js")
        with open(js_path, "w") as fw:
            fw.write("mock js")
        return unittest.mock.Mock(returncode=0)

    with (
        unittest.mock.patch("subprocess.run", new=fake_run),
        unittest.mock.patch("shutil.which", new=fake_which),
    ):
        export_graph(g, out_path, "wasm")

    assert os.path.exists(out_path)


def test_export_mlir_format(tmp_path):
    """Docstring for D103."""
    g = Graph("test")
    g.nodes.append(Node("Add", inputs=[], outputs=["y"]))
    g.nodes.append(Node("Conv", inputs=["y"], outputs=["z"]))
    g.nodes.append(Node("UnknownMLIROp", inputs=["z"], outputs=["u"]))
    out_path = str(tmp_path / "model.mlir")
    export_graph(g, out_path, "mlir")
    assert os.path.exists(out_path)


def test_export_invalid_format(tmp_path):
    """Docstring for D103."""
    g = Graph("test")
    import pytest

    with pytest.raises(ValueError):
        export_graph(g, str(tmp_path / "model.bad"), "bad_format")


def test_export_cpp_format(tmp_path):
    """Docstring for D103."""
    g = Graph("test")
    x = Variable("x", shape=(1,), dtype=DType.FLOAT32)
    y = Variable("y", shape=(1,), dtype=DType.FLOAT32)
    z = Variable("z", shape=(1,), dtype=DType.FLOAT32)
    g.add_tensor(x)
    g.add_tensor(y)
    g.add_tensor(z)
    g.nodes.append(Node("Add", inputs=["x", "y"], outputs=["z"]))
    out_path = str(tmp_path / "model.cpp")
    export_graph(g, out_path, "cpp")
    assert os.path.exists(out_path)


def test_export_onnx_format(tmp_path):
    """Docstring for D103."""
    g = Graph("test")
    x = Variable("x", shape=(1,), dtype=DType.FLOAT32)
    g.add_tensor(x)
    g.nodes.append(Node("Add", inputs=["x", "y"], outputs=["z"]))
    out_path = str(tmp_path / "model.onnx")
    export_graph(g, out_path, "onnx")
    assert os.path.exists(out_path)


def test_generate_keras_fallback():
    """Docstring for D103."""
    g = Graph("test")
    x = Variable("x", shape=(1,), dtype=DType.FLOAT32)
    g.add_tensor(x)
    g.nodes.append(Node("Relu", inputs=["x"], outputs=["z"]))
    g.nodes.append(Node("Add", inputs=["x", "y"], outputs=["z2"]))
    g.nodes.append(Node("BatchNormalization", inputs=["x"], outputs=["z3"]))
    g.nodes.append(Node("Transpose", inputs=["x"], outputs=["z4"]))
    from onnx9000.core.exporter import generate_keras

    code = generate_keras(g)
    assert "ops.relu" in code
    assert "Add()([" in code
    assert "BatchNormalization(axis=1)" in code
    assert "Permute(" in code


def test_generate_mlir():
    """Docstring for D103."""
    g = Graph("test")
    x = Variable("x", shape=(1,), dtype=DType.FLOAT32)
    g.inputs.append("x")
    g.outputs.append("z")
    g.add_tensor(x)
    g.nodes.append(Node("Add", inputs=["x", "y"], outputs=["z"]))
    from onnx9000.core.exporter import generate_mlir

    code = generate_mlir(g)
    assert "func.func @main" in code
    assert "onnx.Add" in code


def test_ir_to_onnx_exporter():
    """Docstring for D103."""
    from onnx9000.core.exporter import IRToONNXExporter

    g = Graph("test")
    g.nodes.append(Node("LayerNorm", inputs=["x"], outputs=["z"]))
    g.nodes.append(Node("CenterCropPad", inputs=["x"], outputs=["z2"]))
    g.nodes.append(Node("CastLike", inputs=["x"], outputs=["z3"]))
    g.nodes.append(Node("IsNaN", inputs=["x"], outputs=["z4"]))
    g.nodes.append(Node("GroupNorm", inputs=["x"], outputs=["z5"]))

    exporter = IRToONNXExporter(opset=21)
    IRToONNXExporter(opset=14)
    # mock export_graph so we don't actually write a file
    import unittest.mock

    with unittest.mock.patch("onnx9000.core.exporter.export_graph"):
        exporter.export(g, "mock.onnx")

    assert g.nodes[0].op_type == "LayerNormalization"
    assert g.nodes[1].op_type == "CenterCropPad"
    assert g.nodes[2].op_type == "CastLike"
    assert g.nodes[3].op_type == "IsNaN"
    assert g.nodes[4].op_type == "GroupNorm"
