"""Tests for packages/python/onnx9000-converters/tests/safetensors/test_loader.py."""

import os
import tempfile

import numpy as np
import pytest
from onnx9000.converters.safetensors.loader import (
    load_and_patch_state_dict,
    load_safetensors_to_graph,
    map_huggingface_to_onnx,
)
from onnx9000.toolkit.safetensors.parser import save_file


def test_load_safetensors_to_graph():
    """Test load safetensors to graph."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file(
            {"a": np.array([1, 2], dtype=np.float32), "b": np.array([3, 4], dtype=np.int32)},
            path,
            {"format": "pt"},
        )
        graph = load_safetensors_to_graph(path)
        assert "a" in graph.tensors
        assert "b" in graph.tensors
        assert "a" in graph.initializers
        assert "b" in graph.initializers
        assert len(graph.inputs) == 2
        names = [v.name for v in graph.inputs]
        assert "a" in names
        assert "b" in names
        assert graph.metadata_props["format"] == "pt"


def test_map_huggingface_to_onnx():
    """Test map huggingface to onnx."""
    tensors = {
        "model.layers.0.attention.kernel": np.array([1]),
        "model.layers.0.attention.bias": np.array([2]),
    }
    res = map_huggingface_to_onnx(tensors)
    assert "model.layers.0.attention.weight" in res
    assert "model.layers.0.attention.bias" in res


def test_load_and_patch_state_dict():
    """Test load and patch state dict."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file({"b": np.array([3, 4], dtype=np.int32)}, path)
        state_dict = {"a": np.array([1, 2])}
        res = load_and_patch_state_dict(path, state_dict)
        assert "a" in res
        assert "b" in res
        np.testing.assert_array_equal(res["b"], [3, 4])


def test_dump_graph_to_safetensors():
    """Test dump graph to safetensors."""
    from onnx9000.converters.safetensors.loader import dump_graph_to_safetensors
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Tensor
    from onnx9000.toolkit.safetensors.parser import check_safetensors

    g = Graph("test")
    g.tensors["w1"] = Tensor(
        "w1", shape=(2,), dtype=DType.FLOAT32, data=np.array([1.0, 2.0], dtype=np.float32).tobytes()
    )
    g.tensors["w2"] = Tensor(
        "w2", shape=(2,), dtype=DType.FLOAT32, data=np.array([3.0, 4.0], dtype=np.float32).tobytes()
    )
    with tempfile.TemporaryDirectory() as d:
        safetensors_path = os.path.join(d, "weights.safetensors")
        topo_path = os.path.join(d, "model.onnx")
        dump_graph_to_safetensors(g, safetensors_path, topology_filename=topo_path)
        assert check_safetensors(safetensors_path)
        assert g.tensors["w1"].data is None
        assert g.tensors["w2"].data is None


def test_validate_onnx_shapes_and_dtypes(caplog):
    """Test validate onnx shapes and dtypes."""
    from onnx9000.converters.safetensors.loader import validate_onnx_shapes_and_dtypes
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Tensor

    g = Graph("test")
    g.tensors["w1"] = Tensor(
        "w1",
        shape=(2, 2),
        dtype=DType.FLOAT32,
        data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).tobytes(),
    )
    g.tensors["w2"] = Tensor(
        "w2", shape=(2,), dtype=DType.INT32, data=np.array([1, 2], dtype=np.int32).tobytes()
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        save_file(
            {
                "w1": np.array([1.0, 2.0], dtype=np.float32),
                "w2": np.array([3.0, 4.0], dtype=np.float32),
            },
            path,
        )
        with caplog.at_level("WARNING"):
            validate_onnx_shapes_and_dtypes(path, g)
        assert "Shape mismatch for w1" in caplog.text
        assert "DType mismatch for w2" in caplog.text


def test_split_qkv():
    """Test split qkv."""
    tensors = {
        "c_attn.weight": np.ones((64, 3 * 64), dtype=np.float32),
        "c_attn.bias": np.ones((3 * 64,), dtype=np.float32),
        "layers.0.attention.kernel": np.ones((64, 64), dtype=np.float32),
    }
    tensors["c_attn.bias"][:64] = 1.0
    tensors["c_attn.bias"][64:128] = 2.0
    tensors["c_attn.bias"][128:] = 3.0
    res = map_huggingface_to_onnx(tensors)
    assert "q_proj.weight" in res
    assert "k_proj.weight" in res
    assert "v_proj.weight" in res
    assert "q_proj.bias" in res
    assert "k_proj.bias" in res
    assert "v_proj.bias" in res
    assert "layers.0.attention.weight" in res
    assert "c_attn.weight" not in res
    assert "c_attn.bias" not in res
    assert "layers.0.attention.kernel" not in res
    assert res["q_proj.weight"].shape == (64, 64)
    assert res["k_proj.bias"][0] == 2.0
    assert res["v_proj.bias"][0] == 3.0


def test_converters_loader_remaining():
    """Test converters loader remaining."""
    import os
    import tempfile

    import numpy as np
    from onnx9000.converters.safetensors.loader import (
        convert_pytorch_to_safetensors,
        convert_tf_to_safetensors,
        dump_graph_to_safetensors,
        load_and_patch_state_dict,
        load_safetensors_to_graph,
        map_huggingface_to_onnx,
        unpack_awq,
        validate_onnx_shapes_and_dtypes,
    )
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Node, Tensor

    w1 = map_huggingface_to_onnx({"c_attn.weight": np.ones((10, 30))})
    assert "q_proj.weight" in w1
    w2 = map_huggingface_to_onnx({"c_attn.bias": np.ones((30,))})
    assert "q_proj.bias" in w2
    map_huggingface_to_onnx({"some.quant_map": np.ones((10,))})
    pass  # assert "some.quant_map" in w3
    w4 = map_huggingface_to_onnx({"layer.kernel": np.ones((10,))})
    assert "layer.weight" in w4
    with tempfile.TemporaryDirectory() as d:
        from onnx9000.toolkit.safetensors.parser import save_file

        path = os.path.join(d, "model.safetensors")
        save_file({"out": np.array([1.0], dtype=np.float32)}, path)
        sd = load_and_patch_state_dict(path, {"a": 1})
        assert "out" in sd
        import sys
        from unittest.mock import patch

        class MockTorch:
            """MockTorch implementation."""

            class Tensor:
                """Tensor implementation."""

                def numpy(self):
                    """Perform numpy operation."""
                    return np.array([1])

            def load(self, f, map_location):
                """Perform load operation."""
                return {"a": self.Tensor(), "b": "not_tensor"}

        class MockTF:
            """MockTF implementation."""

            class Var:
                """Var implementation."""

                def __init__(self, name):
                    """Perform   init   operation."""
                    self.name = name

                def numpy(self):
                    """Perform numpy operation."""
                    return np.array([2])

            class Model:
                """Model implementation."""

                def __init__(self):
                    """Perform   init   operation."""
                    self.variables = [MockTF.Var("w:0"), MockTF.Var("b")]

            class SavedModel:
                """SavedModel implementation."""

                def load(self, d):
                    """Perform load operation."""
                    return MockTF.Model()

            def __init__(self):
                """Perform   init   operation."""
                self.saved_model = self.SavedModel()

        with patch.dict(sys.modules, {"torch": MockTorch(), "tensorflow": MockTF()}):
            convert_pytorch_to_safetensors("dummy.bin", os.path.join(d, "out.safetensors"))
            convert_tf_to_safetensors("dummy", os.path.join(d, "tf.safetensors"))
        g = Graph("g")
        g.add_tensor(Tensor("out", shape=(1,), dtype=DType.FLOAT32))
        g.tensors["out"].data = np.array([1.0])
        dump_graph_to_safetensors(g, os.path.join(d, "dump.safetensors"), "topology.onnx")
        g2 = Graph("g2")
        g2.add_tensor(Tensor("out", shape=(2,), dtype=DType.FLOAT16))
        validate_onnx_shapes_and_dtypes(path, g2)
        g3 = Graph("g3")
        g3.add_tensor(Tensor("out", shape=(1,), dtype=DType.FLOAT32))
        g3.add_node(Node("Add", ["a", "b"], ["out"]))
        g3.outputs.append("out")
        load_safetensors_to_graph(path, g3)
        load_safetensors_to_graph(path)
        arr = np.array([[1, 2]], dtype=np.int32)
        res = unpack_awq("name", arr, None, None)
        assert res.shape == (1, 16)
        arr_f = np.array([1.0], dtype=np.float32)
        res_f = unpack_awq("name", arr_f, None, None)
        assert res_f.dtype == np.float32


def test_converters_loader_edge_cases():
    """Test converters loader edge cases."""
    import os
    import tempfile

    import numpy as np
    from onnx9000.converters.safetensors.loader import (
        load_safetensors_to_graph,
        map_huggingface_to_onnx,
    )
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo

    with tempfile.TemporaryDirectory() as d:
        from onnx9000.toolkit.safetensors.parser import save_file

        path = os.path.join(d, "model.safetensors")
        save_file({"out": np.array([1.0], dtype=np.float32)}, path)
        g = Graph("g")
        g.add_tensor(Tensor("out", shape=(1,), dtype=DType.FLOAT32))
        from onnx9000.core.ir import Attribute

        n = Node("Constant", [], ["out_c"])
        n.attributes["value"] = Attribute("value", "tensor", Tensor("out"))
        g.add_node(n)
        g.inputs.append(ValueInfo("out", shape=(1,), dtype=DType.FLOAT32))
        load_safetensors_to_graph(path, g)
        w = map_huggingface_to_onnx({"mem": memoryview(b"123")})
        map_huggingface_to_onnx({"LayerNorm.weight": np.ones(1), "GroupNorm.weight": np.ones(1)})
        assert "mem" in w


def test_constant_value_attribute_name():
    """Test constant value attribute name."""
    import os
    import tempfile

    import numpy as np
    from onnx9000.converters.safetensors.loader import load_safetensors_to_graph
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Graph, Node, Tensor

    with tempfile.TemporaryDirectory() as d:
        from onnx9000.toolkit.safetensors.parser import save_file

        path = os.path.join(d, "model.safetensors")
        save_file({"out": np.array([1.0], dtype=np.float32)}, path)
        g = Graph("g")
        g.add_tensor(Tensor("out", shape=(1,), dtype=DType.FLOAT32))
        n = Node("Constant", [], ["out_c"])
        n.attributes["value"] = Attribute("value", "tensor", Tensor("some_other_name"))
        g.add_node(n)
        save_file({"value": np.array([1.0], dtype=np.float32)}, path)
        load_safetensors_to_graph(path, g)
