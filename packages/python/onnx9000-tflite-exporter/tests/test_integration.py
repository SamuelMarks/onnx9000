"""Tests for packages/python/onnx9000-tflite-exporter/tests/test_integration.py."""

import struct
import time

import pytest
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.tflite_exporter.compiler.layout import LayoutOptimizer
from onnx9000.tflite_exporter.compiler.subgraph import compile_graph_to_tflite
from onnx9000.tflite_exporter.exporter import TFLiteExporter


def test_performance():
    """Test performance."""
    graph = Graph("MassiveDummy")
    graph.inputs.append(ValueInfo("X", (1, 3, 224, 224), "float32"))
    num_elements = 10 * 1024 * 1024
    w_data = struct.pack(f"<{num_elements}f", *[1.0] * num_elements)
    graph.tensors["W"] = Tensor(
        "W", shape=(num_elements,), dtype="float32", is_initializer=True, data=w_data
    )
    graph.nodes.append(Node("Add", ["X", "W"], ["Y"], name="add_massive"))
    graph.outputs.append(ValueInfo("Y", (1, 3, 224, 224), "float32"))
    graph.tensors["X"] = Tensor("X", shape=(1, 3, 224, 224), dtype="float32", is_initializer=False)
    graph.tensors["Y"] = Tensor("Y", shape=(1, 3, 224, 224), dtype="float32", is_initializer=False)
    start = time.time()
    exporter = TFLiteExporter()
    layout_opt = LayoutOptimizer(graph, keep_nchw=False)
    layout_opt.optimize()
    subgraphs_offset = compile_graph_to_tflite(graph, exporter, keep_nchw=False)
    exporter.builder.start_vector(4, 1, 4)
    exporter.builder.add_offset(subgraphs_offset)
    subgraphs_vec_offset = exporter.builder.end_vector(1)
    buf = exporter.finish(subgraphs_vec_offset, "onnx9000_massive")
    end = time.time()
    diff_ms = (end - start) * 1000
    assert diff_ms < 5000
    assert len(buf) > 10 * 1024 * 1024


def test_resnet50_topology():
    """Test resnet50 topology."""
    graph = Graph("ResNet50")
    graph.inputs.append(ValueInfo("Input", (1, 3, 224, 224), "float32"))
    graph.tensors["Input"] = Tensor(
        "Input", shape=(1, 3, 224, 224), dtype="float32", is_initializer=False
    )
    w_data = struct.pack(f"<{64 * 3 * 7 * 7}f", *[1.0] * (64 * 3 * 7 * 7))
    graph.tensors["Conv1_W"] = Tensor(
        "Conv1_W", shape=(64, 3, 7, 7), dtype="float32", is_initializer=True, data=w_data
    )
    from onnx9000.core.ir import Attribute

    graph.nodes.append(
        Node(
            "Conv",
            ["Input", "Conv1_W"],
            ["Conv1_Out"],
            {
                "strides": Attribute("strides", "INTS", [2, 2]),
                "pads": Attribute("pads", "INTS", [3, 3, 3, 3]),
            },
            "conv1",
        )
    )
    graph.tensors["Conv1_Out"] = Tensor(
        "Conv1_Out", shape=(1, 64, 112, 112), dtype="float32", is_initializer=False
    )
    graph.nodes.append(Node("Relu", ["Conv1_Out"], ["Relu1_Out"], {}, "relu1"))
    graph.tensors["Relu1_Out"] = Tensor(
        "Relu1_Out", shape=(1, 64, 112, 112), dtype="float32", is_initializer=False
    )
    graph.nodes.append(
        Node(
            "MaxPool",
            ["Relu1_Out"],
            ["Pool1_Out"],
            {
                "kernel_shape": Attribute("kernel_shape", "INTS", [3, 3]),
                "strides": Attribute("strides", "INTS", [2, 2]),
                "pads": Attribute("pads", "INTS", [1, 1, 1, 1]),
            },
            "pool1",
        )
    )
    graph.tensors["Pool1_Out"] = Tensor(
        "Pool1_Out", shape=(1, 64, 56, 56), dtype="float32", is_initializer=False
    )
    graph.nodes.append(Node("GlobalAveragePool", ["Pool1_Out"], ["Gap_Out"], {}, "gap"))
    graph.tensors["Gap_Out"] = Tensor(
        "Gap_Out", shape=(1, 64, 1, 1), dtype="float32", is_initializer=False
    )
    import numpy as np

    shape_data = np.array([1, 64], dtype=np.int64).tobytes()
    graph.tensors["Reshape_Shape"] = Tensor(
        "Reshape_Shape", shape=(2,), dtype="int64", is_initializer=True, data=shape_data
    )
    graph.nodes.append(
        Node("Reshape", ["Gap_Out", "Reshape_Shape"], ["Reshape_Out"], {}, "reshape")
    )
    graph.tensors["Reshape_Out"] = Tensor(
        "Reshape_Out", shape=(1, 64), dtype="float32", is_initializer=False
    )
    fc_data = struct.pack(f"<{1000 * 64}f", *[1.0] * (1000 * 64))
    graph.tensors["FC_W"] = Tensor(
        "FC_W", shape=(1000, 64), dtype="float32", is_initializer=True, data=fc_data
    )
    graph.nodes.append(Node("Gemm", ["Reshape_Out", "FC_W"], ["Output"], {}, "gemm"))
    graph.tensors["Output"] = Tensor(
        "Output", shape=(1, 1000), dtype="float32", is_initializer=False
    )
    graph.outputs.append(ValueInfo("Output", (1, 1000), "float32"))
    exporter = TFLiteExporter()
    layout_opt = LayoutOptimizer(graph, keep_nchw=False)
    layout_opt.optimize()
    subgraphs_offset = compile_graph_to_tflite(graph, exporter, keep_nchw=False)
    exporter.builder.start_vector(4, 1, 4)
    exporter.builder.add_offset(subgraphs_offset)
    subgraphs_vec_offset = exporter.builder.end_vector(1)
    buf = exporter.finish(subgraphs_vec_offset, "resnet50_mock")
    assert len(buf) > 10000


def test_mobilenet_v2_topology():
    """Test mobilenet v2 topology."""
    graph = Graph("MobileNetV2")
    graph.inputs.append(ValueInfo("Input", (1, 3, 224, 224), "float32"))
    graph.tensors["Input"] = Tensor(
        "Input", shape=(1, 3, 224, 224), dtype="float32", is_initializer=False
    )
    from onnx9000.core.ir import Attribute

    w_data1 = struct.pack(f"<{32 * 27}f", *[1.0] * (32 * 27))
    graph.tensors["Conv1_W"] = Tensor(
        "Conv1_W", shape=(32, 3, 3, 3), dtype="float32", is_initializer=True, data=w_data1
    )
    graph.nodes.append(
        Node(
            "Conv",
            ["Input", "Conv1_W"],
            ["Conv1_Out"],
            {
                "strides": Attribute("strides", "INTS", [2, 2]),
                "pads": Attribute("pads", "INTS", [1, 1, 1, 1]),
            },
            "conv1",
        )
    )
    graph.tensors["Conv1_Out"] = Tensor(
        "Conv1_Out", shape=(1, 32, 112, 112), dtype="float32", is_initializer=False
    )
    graph.nodes.append(Node("Relu", ["Conv1_Out"], ["Relu1_Out"], {}, "relu1"))
    graph.tensors["Relu1_Out"] = Tensor(
        "Relu1_Out", shape=(1, 32, 112, 112), dtype="float32", is_initializer=False
    )
    dw_data = struct.pack(f"<{32 * 9}f", *[1.0] * (32 * 9))
    graph.tensors["DW_W"] = Tensor(
        "DW_W", shape=(32, 1, 3, 3), dtype="float32", is_initializer=True, data=dw_data
    )
    graph.nodes.append(
        Node(
            "Conv",
            ["Relu1_Out", "DW_W"],
            ["DW_Out"],
            {
                "strides": Attribute("strides", "INTS", [1, 1]),
                "pads": Attribute("pads", "INTS", [1, 1, 1, 1]),
                "group": Attribute("group", "INT", 32),
            },
            "dw_conv",
        )
    )
    graph.tensors["DW_Out"] = Tensor(
        "DW_Out", shape=(1, 32, 112, 112), dtype="float32", is_initializer=False
    )
    graph.nodes.append(Node("Relu6", ["DW_Out"], ["Relu6_Out"], {}, "relu6"))
    graph.tensors["Relu6_Out"] = Tensor(
        "Relu6_Out", shape=(1, 32, 112, 112), dtype="float32", is_initializer=False
    )
    pw_data = struct.pack(f"<{16 * 32}f", *[1.0] * (16 * 32))
    graph.tensors["PW_W"] = Tensor(
        "PW_W", shape=(16, 32, 1, 1), dtype="float32", is_initializer=True, data=pw_data
    )
    graph.nodes.append(
        Node(
            "Conv",
            ["Relu6_Out", "PW_W"],
            ["PW_Out"],
            {"strides": Attribute("strides", "INTS", [1, 1])},
            "pw_conv",
        )
    )
    graph.tensors["PW_Out"] = Tensor(
        "PW_Out", shape=(1, 16, 112, 112), dtype="float32", is_initializer=False
    )
    graph.outputs.append(ValueInfo("PW_Out", (1, 16, 112, 112), "float32"))
    exporter = TFLiteExporter()
    layout_opt = LayoutOptimizer(graph, keep_nchw=False)
    layout_opt.optimize()
    subgraphs_offset = compile_graph_to_tflite(graph, exporter, keep_nchw=False)
    exporter.builder.start_vector(4, 1, 4)
    exporter.builder.add_offset(subgraphs_offset)
    subgraphs_vec_offset = exporter.builder.end_vector(1)
    buf = exporter.finish(subgraphs_vec_offset, "mobilenetv2_mock")
    assert len(buf) > 1000


def test_yolov8_topology():
    """Test yolov8 topology."""
    graph = Graph("YOLOv8")
    graph.inputs.append(ValueInfo("images", (1, 3, 640, 640), "float32"))
    graph.tensors["images"] = Tensor(
        "images", shape=(1, 3, 640, 640), dtype="float32", is_initializer=False
    )
    from onnx9000.core.ir import Attribute

    w_data1 = struct.pack(f"<{16 * 27}f", *[1.0] * (16 * 27))
    graph.tensors["Conv1_W"] = Tensor(
        "Conv1_W", shape=(16, 3, 3, 3), dtype="float32", is_initializer=True, data=w_data1
    )
    graph.nodes.append(
        Node(
            "Conv",
            ["images", "Conv1_W"],
            ["Conv1_Out"],
            {
                "strides": Attribute("strides", "INTS", [2, 2]),
                "pads": Attribute("pads", "INTS", [1, 1, 1, 1]),
            },
            "conv1",
        )
    )
    graph.tensors["Conv1_Out"] = Tensor(
        "Conv1_Out", shape=(1, 16, 320, 320), dtype="float32", is_initializer=False
    )
    import numpy as np

    split_data = np.array([8, 8], dtype=np.int64).tobytes()
    graph.tensors["Split_Size"] = Tensor(
        "Split_Size", shape=(2,), dtype="int64", is_initializer=True, data=split_data
    )
    graph.nodes.append(
        Node(
            "Split",
            ["Conv1_Out", "Split_Size"],
            ["Split1", "Split2"],
            {"axis": Attribute("axis", "INT", 1)},
            "split",
        )
    )
    graph.tensors["Split1"] = Tensor(
        "Split1", shape=(1, 8, 320, 320), dtype="float32", is_initializer=False
    )
    graph.tensors["Split2"] = Tensor(
        "Split2", shape=(1, 8, 320, 320), dtype="float32", is_initializer=False
    )
    graph.nodes.append(Node("Add", ["Split1", "Split2"], ["Add_Out"], {}, "add_residual"))
    graph.tensors["Add_Out"] = Tensor(
        "Add_Out", shape=(1, 8, 320, 320), dtype="float32", is_initializer=False
    )
    graph.nodes.append(
        Node(
            "Concat",
            ["Split1", "Split2", "Add_Out"],
            ["Concat_Out"],
            {"axis": Attribute("axis", "INT", 1)},
            "concat",
        )
    )
    graph.tensors["Concat_Out"] = Tensor(
        "Concat_Out", shape=(1, 24, 320, 320), dtype="float32", is_initializer=False
    )
    head_data = struct.pack(f"<{84 * 24}f", *[1.0] * (84 * 24))
    graph.tensors["Head_W"] = Tensor(
        "Head_W", shape=(84, 24, 1, 1), dtype="float32", is_initializer=True, data=head_data
    )
    graph.nodes.append(Node("Conv", ["Concat_Out", "Head_W"], ["Head_Out"], {}, "head_conv"))
    graph.tensors["Head_Out"] = Tensor(
        "Head_Out", shape=(1, 84, 8400), dtype="float32", is_initializer=False
    )
    graph.outputs.append(ValueInfo("Head_Out", (1, 84, 8400), "float32"))
    exporter = TFLiteExporter()
    layout_opt = LayoutOptimizer(graph, keep_nchw=False)
    layout_opt.optimize()
    subgraphs_offset = compile_graph_to_tflite(graph, exporter, keep_nchw=False)
    exporter.builder.start_vector(4, 1, 4)
    exporter.builder.add_offset(subgraphs_offset)
    subgraphs_vec_offset = exporter.builder.end_vector(1)
    buf = exporter.finish(subgraphs_vec_offset, "yolov8_mock")
    assert len(buf) > 1000


def test_whisper_topology():
    """Test whisper topology."""
    graph = Graph("Whisper")
    graph.inputs.append(ValueInfo("mels", (1, 80, 3000), "float32"))
    graph.tensors["mels"] = Tensor(
        "mels", shape=(1, 80, 3000), dtype="float32", is_initializer=False
    )
    from onnx9000.core.ir import Attribute

    w_data1 = struct.pack(f"<{384 * 80 * 3}f", *[1.0] * (384 * 80 * 3))
    graph.tensors["Conv1_W"] = Tensor(
        "Conv1_W", shape=(384, 80, 3), dtype="float32", is_initializer=True, data=w_data1
    )
    graph.nodes.append(
        Node(
            "Conv",
            ["mels", "Conv1_W"],
            ["Conv1_Out"],
            {
                "strides": Attribute("strides", "INTS", [1]),
                "pads": Attribute("pads", "INTS", [1, 1]),
            },
            "conv1d_feat",
        )
    )
    graph.tensors["Conv1_Out"] = Tensor(
        "Conv1_Out", shape=(1, 384, 3000), dtype="float32", is_initializer=False
    )
    graph.nodes.append(Node("Gelu", ["Conv1_Out"], ["Gelu_Out"], {}, "gelu1"))
    graph.tensors["Gelu_Out"] = Tensor(
        "Gelu_Out", shape=(1, 384, 3000), dtype="float32", is_initializer=False
    )
    graph.nodes.append(
        Node(
            "Transpose",
            ["Gelu_Out"],
            ["Trans_Out"],
            {"perm": Attribute("perm", "INTS", [0, 2, 1])},
            "trans_attn",
        )
    )
    graph.tensors["Trans_Out"] = Tensor(
        "Trans_Out", shape=(1, 3000, 384), dtype="float32", is_initializer=False
    )
    qkv_data = struct.pack(f"<{384 * 1152}f", *[1.0] * (384 * 1152))
    graph.tensors["QKV_W"] = Tensor(
        "QKV_W", shape=(384, 1152), dtype="float32", is_initializer=True, data=qkv_data
    )
    graph.nodes.append(Node("MatMul", ["Trans_Out", "QKV_W"], ["QKV_Out"], {}, "matmul_attn"))
    graph.tensors["QKV_Out"] = Tensor(
        "QKV_Out", shape=(1, 3000, 1152), dtype="float32", is_initializer=False
    )
    graph.outputs.append(ValueInfo("QKV_Out", (1, 3000, 1152), "float32"))
    exporter = TFLiteExporter()
    layout_opt = LayoutOptimizer(graph, keep_nchw=False)
    layout_opt.optimize()
    subgraphs_offset = compile_graph_to_tflite(graph, exporter, keep_nchw=False)
    exporter.builder.start_vector(4, 1, 4)
    exporter.builder.add_offset(subgraphs_offset)
    subgraphs_vec_offset = exporter.builder.end_vector(1)
    buf = exporter.finish(subgraphs_vec_offset, "whisper_mock")
    assert len(buf) > 1000


def test_deeplabv3_topology():
    """Test deeplabv3 topology."""
    graph = Graph("DeepLabV3")
    graph.inputs.append(ValueInfo("Image", (1, 3, 513, 513), "float32"))
    graph.tensors["Image"] = Tensor(
        "Image", shape=(1, 3, 513, 513), dtype="float32", is_initializer=False
    )
    from onnx9000.core.ir import Attribute

    feat_data = struct.pack(f"<{256 * 27}f", *[1.0] * (256 * 27))
    graph.tensors["Feat_W"] = Tensor(
        "Feat_W", shape=(256, 3, 3, 3), dtype="float32", is_initializer=True, data=feat_data
    )
    graph.nodes.append(
        Node(
            "Conv",
            ["Image", "Feat_W"],
            ["Feat_Out"],
            {"strides": Attribute("strides", "INTS", [2, 2])},
            "backbone_conv",
        )
    )
    graph.tensors["Feat_Out"] = Tensor(
        "Feat_Out", shape=(1, 256, 257, 257), dtype="float32", is_initializer=False
    )
    aspp1_data = struct.pack(f"<{256 * 256 * 9}f", *[1.0] * (256 * 256 * 9))
    graph.tensors["ASPP1_W"] = Tensor(
        "ASPP1_W", shape=(256, 256, 3, 3), dtype="float32", is_initializer=True, data=aspp1_data
    )
    graph.nodes.append(
        Node(
            "Conv",
            ["Feat_Out", "ASPP1_W"],
            ["Branch1_Out"],
            {
                "dilations": Attribute("dilations", "INTS", [6, 6]),
                "pads": Attribute("pads", "INTS", [6, 6, 6, 6]),
            },
            "aspp1",
        )
    )
    graph.tensors["Branch1_Out"] = Tensor(
        "Branch1_Out", shape=(1, 256, 257, 257), dtype="float32", is_initializer=False
    )
    graph.nodes.append(Node("GlobalAveragePool", ["Feat_Out"], ["Pool_Out"], {}, "aspp_pool"))
    graph.tensors["Pool_Out"] = Tensor(
        "Pool_Out", shape=(1, 256, 1, 1), dtype="float32", is_initializer=False
    )
    import numpy as np

    resize_data = np.array([1, 256, 257, 257], dtype=np.int64).tobytes()
    graph.tensors["Resize_Shape"] = Tensor(
        "Resize_Shape", shape=(4,), dtype="int64", is_initializer=True, data=resize_data
    )
    graph.nodes.append(
        Node(
            "Resize",
            ["Pool_Out", "", "Resize_Shape"],
            ["Branch2_Out"],
            {"mode": Attribute("mode", "STRING", "nearest")},
            "aspp_resize",
        )
    )
    graph.tensors["Branch2_Out"] = Tensor(
        "Branch2_Out", shape=(1, 256, 257, 257), dtype="float32", is_initializer=False
    )
    graph.nodes.append(
        Node(
            "Concat",
            ["Branch1_Out", "Branch2_Out"],
            ["Concat_Out"],
            {"axis": Attribute("axis", "INT", 1)},
            "concat_aspp",
        )
    )
    graph.tensors["Concat_Out"] = Tensor(
        "Concat_Out", shape=(1, 512, 257, 257), dtype="float32", is_initializer=False
    )
    graph.outputs.append(ValueInfo("Concat_Out", (1, 512, 257, 257), "float32"))
    exporter = TFLiteExporter()
    layout_opt = LayoutOptimizer(graph, keep_nchw=False)
    layout_opt.optimize()
    subgraphs_offset = compile_graph_to_tflite(graph, exporter, keep_nchw=False)
    exporter.builder.start_vector(4, 1, 4)
    exporter.builder.add_offset(subgraphs_offset)
    subgraphs_vec_offset = exporter.builder.end_vector(1)
    buf = exporter.finish(subgraphs_vec_offset, "deeplabv3_mock")
    assert len(buf) > 1000
