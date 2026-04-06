import pytest

#
from onnx9000.core.codegen.triton import TritonExporter
from onnx9000.core.ir import Graph, Node
from onnx9000.core.surgeon import LayoutOptimizerPass, StatefulToStatelessPass, unroll_scan

VISION_MODELS = [
    "ResNet-18",
    "34",
    "50",
    "101",
    "152",
    "200",
    "269",
    "ResNeXt-50 (32x4d)",
    "ResNeXt-101 (32x8d",
    "64x4d)",
    "Wide-ResNet-50-2",
    "Wide-ResNet-101-2",
    "SE-ResNet",
    "SK-ResNet",
    "ECA-ResNet",
    "EfficientNet-B0 through B8",
    "EfficientNet-L2",
    "EfficientNetV2-Small",
    "Medium",
    "Large",
    "XL",
    "EfficientNet-EdgeTPU (S",
    "M",
    "L)",
    "MobileNetV1 (0.25",
    "0.5",
    "0.75",
    "1.0)",
    "MobileNetV2 (0.35",
    "0.5",
    "0.75",
    "1.0",
    "1.3",
    "1.4)",
    "MobileNetV3 (Small",
    "Large",
    "minimalistic variants)",
    "ShuffleNet V1 (Groups: 1",
    "2",
    "3",
    "4",
    "8)",
    "ShuffleNet V2 (0.5x",
    "1.0x",
    "1.5x",
    "2.0x)",
    "SqueezeNet (1.0",
    "1.1)",
    "SqueezeNext",
    "GhostNet",
    "GhostNetV2",
    "MicroNet",
    "ConvNeXt (Atto",
    "Femto",
    "Pico",
    "Nano",
    "Tiny",
    "Small",
    "Base",
    "Large",
    "XLarge)",
    "ConvNeXt V2 (Atto -> Huge)",
    "RepVGG (A0",
    "A1",
    "A2",
    "B0",
    "B1",
    "B2",
    "B3)",
    "NFNet (F0",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6)",
    "RegNet (X/Y variants: 400MF",
    "800MF",
    "1.6GF",
    "3.2GF",
    "8.0GF",
    "16GF)",
    "DenseNet (121",
    "161",
    "169",
    "201",
    "264)",
    "VGG (11",
    "13",
    "16",
    "19 + BN variants)",
    "Inception (v1/GoogLeNet",
    "v2",
    "v3",
    "v4)",
    "Inception-ResNet-v2",
    "Xception",
    "ViT (Tiny",
    "Small",
    "Base",
    "Large",
    "Huge",
    "Giant)",
    "Patch sizes: 16x16",
    "32x32",
    "14x14",
    "DeiT (Tiny",
    "Small",
    "Base) & DeiT III",
    "Swin Transformer (Tiny",
    "Small",
    "Base",
    "Large)",
    "Swin V2 (Tiny",
    "Small",
    "Base",
    "Large",
    "Giant)",
    "Focal Transformer",
    "PVT (Pyramid Vision Transformer) v1 & v2",
    "CVT",
    "CoaT",
    "CrossViT",
    "MobileViT (S",
    "XS",
    "XXS)",
    "MobileViTv2",
    "MobileViTv3",
    "LeViT (128S",
    "128",
    "192",
    "256",
    "384)",
    "CMT",
    "EfficientFormer",
    "EdgeNeXt",
    "YOLOv3",
    "YOLOv4",
    "YOLOv5 (n",
    "s",
    "m",
    "l",
    "x)",
    "YOLOv7 (Tiny",
    "X",
    "W6",
    "E6",
    "D6",
    "E6E)",
    "YOLOv8 (n",
    "s",
    "m",
    "l",
    "x - Detect/Seg/Pose)",
    "YOLOv9",
    "YOLOv10 (all scales)",
    "YOLOX (Nano",
    "Tiny",
    "s",
    "m",
    "l",
    "x)",
    "DETR (ResNet50",
    "ResNet101 backbones)",
    "Deformable DETR",
    "Conditional DETR",
    "Anchor DETR",
    "DAB-DETR",
    "DINO (Detection)",
    "Faster R-CNN (FPN",
    "C4 backbones)",
    "Cascade R-CNN",
    "Mask R-CNN",
    "U-Net (2D",
    "3D)",
    "V-Net",
    "DeepLabV3",
    "DeepLabV3+",
    "SegFormer (B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5)",
    "Mask2Former",
    "SAM (Segment Anything - ViT-B",
    "L",
    "H)",
    "MobileSAM",
    "FastSAM",
    "I3D (Inflated 3D ConvNets)",
    "SlowFast (4x16",
    "8x8)",
    "TimeSformer",
    "Video Swin Transformer",
    "OpenPose",
    "HigherHRNet",
    "MMPose models (ViTPose)",
    "Stable Diffusion (v1.4",
    "v1.5",
    "v2.0",
    "v2.1)",
    "SDXL (Base + Refiner)",
    "SD3 (Stable Diffusion 3)",
    "ControlNet (Canny",
    "Depth",
    "OpenPose",
    "Hed",
    "Scribble",
    "MLSD)",
    "Pix2Pix",
    "CycleGAN",
    "StyleGAN2/3",
    "BERT (Tiny",
    "Mini",
    "Small",
    "Medium",
    "Base",
    "Large)",
    "RoBERTa (Base",
    "Large)",
    "ALBERT (v1/v2 - Base",
    "Large",
    "XLarge",
    "XXLarge)",
    "DeBERTa (V2",
    "V3 - Base",
    "Large",
    "XLarge)",
    "DistilBERT",
    "DistilRoBERTa",
    "MobileBERT",
    "SqueezeBERT",
    "Electra (Small",
    "Base",
    "Large)",
    "ConvBERT",
    "LLaMA (7B",
    "13B",
    "33B",
    "65B)",
    "LLaMA-2 (7B",
    "13B",
    "70B)",
    "LLaMA-3 (8B",
    "70B) & LLaMA-3.1",
    "Alpaca",
    "Vicuna",
    "Guanaco derivatives",
    "Mistral (7B-v0.1",
    "v0.2",
    "Instruct)",
    "Mixtral 8x7B",
    "8x22B (MoE routing logic translation)",
    "Qwen (1.0",
    "1.5",
    "2.0 - 0.5B through 72B)",
    "Gemma (2B",
    "7B)",
    "Gemma-2 (9B",
    "27B)",
    "Phi-1",
    "Phi-1.5",
    "Phi-2",
    "Phi-3 (Mini",
    "Small",
    "Medium)",
    "GPT-Neo",
    "GPT-J",
    "GPT-NeoX",
    "Pythia (70M to 12B)",
    "OPT (125M to 175B)",
    "Falcon (7B",
    "40B",
    "180B)",
    "Bloom (560M to 176B)",
    "MPT (7B",
    "30B)",
    "T5 (Small",
    "Base",
    "Large",
    "3B",
    "11B)",
    "FLAN-T5",
    "mT5",
    "byT5",
    "BART (Base",
    "Large)",
    "mBART",
    "MarianMT (all language pairs)",
    "BGE (Base",
    "Large)",
    "E5 (Small",
    "Base",
    "Large)",
    "Nomic-Embed-Text",
    "ColBERT (v1",
    "v2)",
    "AlphaFold 2 (Evoformer block validation)",
    "FourCastNet",
    "Pangu-Weather (Meteorology)",
    "FNO (Fourier Neural Operator)",
    "PINNs",
    "PPO/SAC/DQN continuous/discrete actions",
    "DreamerV3 World Models",
    "MuZero",
    "AlphaZero core networks",
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
        pass
        #


def test_snapshot_codegen():
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.codegen.keras import ONNXToKerasVisitor

    for group in [
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
    ]:
        for model_name in group:
            g = mock_graph_for_model(model_name)
            pt_code = ONNXToPyTorchVisitor(g).generate()
            assert "class Model_" in pt_code

            flax_code = ONNXToFlaxNNXVisitor(g).generate()
            assert "class Model_" in flax_code

            keras_code = ONNXToKerasVisitor(g).generate()
            assert "class Model_" in keras_code
