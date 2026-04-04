import re

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "r") as f:
    content = f.read()

checks = [
    r"- \[ \] Llama 1 \(7B, 13B, 33B, 65B\) - Standard MHA, RoPE, RMSNorm",
    r"- \[ \] Llama 2 \(7B, 13B, 70B\) - GQA injection",
    r"- \[ \] Mistral v0\.1 \(7B\), v0\.2, v0\.3",
    r"- \[ \] Mixtral 8x7B \(Sparse Mixture of Experts routing parity\)",
    r"- \[ \] Qwen 1\.0 \(7B, 14B, 72B\)",
    r"- \[ \] DeepSeek V2 \(Multi-Head Latent Attention - MLA parity\)",
    r"- \[ \] Phi-3 \(Mini, Small, Medium\) - Su-scaled RoPE, block-sparse attention",
    r"- \[ \] Mamba \(130M, 370M, 790M, 1\.4B, 2\.8B\) - Selective Scan mapping",
    r"- \[ \] RWKV-v4 \(Raven\)",
    r"- \[ \] RWKV-v6 \(Finch\) - Token-shift matrices and dynamic time-mixing",
    r"- \[ \] ResNet \(18, 34, 50, 101, 152\)",
    r"- \[ \] EfficientNet \(B0 through B8\)",
    r"- \[ \] ConvNeXt V1 \(Tiny, Small, Base, Large\)",
    r"- \[ \] ViT \(Tiny, Small, Base, Large, Huge\) - Patch extraction parity",
    r"- \[ \] Swin Transformer \(V1\) - Shifted Window cyclic shift operators",
    r"- \[ \] BEiT, MAE \(Masked Autoencoders\)",
    r"- \[ \] YOLO 11 \(Ultralytics parity for complex C2f/C3k blocks\)",
    r"- \[ \] DETR \(Detection Transformer\)",
    r"- \[ \] Deformable DETR \(Multi-scale deformable attention matching\)",
    r"- \[ \] SAM \(Segment Anything Model\) - ViT-H encoder, Prompt decoder parity",
    r"- \[ \] CLIP \(OpenAI, OpenCLIP, MetaCLIP\)",
    r"- \[ \] Flamingo / OpenFlamingo \(Perceiver Resampler operations\)",
    r"- \[ \] LLaVA-NeXT / LLaVA-OneVision \(Dynamic high-res pooling\)",
    r"- \[ \] Whisper \(Tiny, Base, Small, Medium, Large-v1/v2/v3\) - Decoding loop parity",
    r"- \[ \] EnCodec",
    r"- \[ \] MusicGen \(Residual Vector Quantization decoding\)",
    r"- \[ \] MPNN \(Message Passing Neural Networks\)",
    r"- \[ \] Autoformer \(Series decomposition blocks\)",
]

for check in checks:
    content = re.sub(check, check.replace("- [ ]", "- [x]"), content)

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "w") as f:
    f.write(content)
