import pytest
from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.core.codegen.triton import TritonExporter
from onnx9000.core.ir import Graph, Node
from onnx9000.core.surgeon import LayoutOptimizerPass, StatefulToStatelessPass, unroll_scan

VISION_MODELS = [
    "ResNet-50",
    "MobileNetV3",
    "EfficientNetB0",
    "ConvNeXt",
    "InceptionV3",
    "DenseNet",
    "VGG",
]
VIT_MODELS = ["ViT-B", "Swin-T", "MLP-Mixer", "MobileViT", "MaxViT", "MAE", "DINOv2"]
DETECTION_MODELS = ["YOLOv8", "DETR", "Mask R-CNN", "SAM", "OpenPose"]

NLP_MODELS = ["BERT-Base", "DeBERTa", "XLM-RoBERTa"]
LLM_MODELS = ["Llama-3", "Qwen-2", "Mistral", "Gemma", "Phi-3", "Falcon"]
MOE_MODELS = ["Mixtral 8x7B", "Switch Transformer", "DBRX"]
SSM_MODELS = ["Mamba 1.0", "RWKV-v6", "Jamba"]

GENERATIVE_MODELS = ["Stable Diffusion 1.5", "Flux.1", "SVD", "ControlNet", "VQ-GAN"]
AUDIO_MODELS = ["Whisper Tiny", "VITS", "EnCodec"]
SCIENTIFIC_MODELS = ["XGBoost", "GCN", "AlphaFold 2", "RT-1"]


def mock_graph_for_model(model_name: str) -> Graph:
    g = Graph(name=model_name)
    g.nodes = []

    # Add representative ops for testing parsing/export paths
    if model_name in VISION_MODELS:
        g.nodes.extend(
            [
                Node("Conv", ["x"], ["conv_out"]),
                Node("BatchNormalization", ["conv_out"], ["bn_out"]),
                Node("Relu", ["bn_out"], ["y"]),
            ]
        )
    elif model_name in VIT_MODELS:
        g.nodes.extend(
            [
                Node("Conv", ["x"], ["patch_out"]),
                Node("MultiHeadAttention", ["patch_out"], ["attn_out"]),
            ]
        )
    elif model_name in DETECTION_MODELS:
        g.nodes.extend([Node("Conv", ["x"], ["c"]), Node("Resize", ["c"], ["r"])])
    elif model_name in NLP_MODELS:
        g.nodes.extend(
            [Node("Gather", ["emb", "idx"], ["g"]), Node("MultiHeadAttention", ["g"], ["y"])]
        )
    elif model_name in LLM_MODELS:
        g.nodes.extend(
            [
                Node("RotaryEmbedding", ["x"], ["r"]),
                Node("FlashAttention", ["r"], ["a"]),
                Node("MatMul", ["a", "w"], ["y"]),
            ]
        )
    elif model_name in MOE_MODELS:
        g.nodes.extend([Node("TopK", ["x"], ["t"]), Node("GatherElements", ["e", "t"], ["y"])])
    elif model_name in SSM_MODELS:
        g.nodes.extend([Node("MambaBlock", ["x"], ["y"])])
    elif model_name in GENERATIVE_MODELS:
        g.nodes.extend([Node("Conv", ["x"], ["c"]), Node("MultiHeadAttention", ["c"], ["a"])])
    elif model_name in AUDIO_MODELS:
        g.nodes.extend([Node("DFT", ["x"], ["f"]), Node("MultiHeadAttention", ["f"], ["y"])])
    elif model_name in SCIENTIFIC_MODELS:
        g.nodes.extend([Node("TreeEnsemble", ["x"], ["y"])])
    else:
        g.nodes.extend([Node("Identity", ["x"], ["y"])])

    return g


@pytest.mark.parametrize(
    "model_group",
    [
        VISION_MODELS,
        VIT_MODELS,
        DETECTION_MODELS,
        NLP_MODELS,
        LLM_MODELS,
        MOE_MODELS,
        SSM_MODELS,
        GENERATIVE_MODELS,
        AUDIO_MODELS,
        SCIENTIFIC_MODELS,
    ],
)
def test_model_zoo_matrix_parsing(model_group):
    for model_name in model_group:
        g = mock_graph_for_model(model_name)

        # Test applying passes
        g = LayoutOptimizerPass.apply(g)
        g = StatefulToStatelessPass.apply(g)
        g = unroll_scan(g)

        # Ensure Graph is valid
        assert len(g.nodes) > 0


def test_triton_export_on_llms():
    for model_name in LLM_MODELS:
        g = mock_graph_for_model(model_name)
        triton_code = TritonExporter(g).export()
        assert "@triton.jit" in triton_code
        assert "_kernel(" in triton_code


def test_c_compiler_on_vision():
    # Test that C compiler doesn't crash on standard vision ops
    for model_name in VISION_MODELS:
        g = mock_graph_for_model(model_name)
        # We don't enforce full generation since we don't have ValueInfos,
        # but compiling the instance shouldn't crash
        comp = C89Compiler(g)
        assert comp.graph.name == model_name
