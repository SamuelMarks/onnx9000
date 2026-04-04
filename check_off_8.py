import re

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "r") as f:
    content = f.read()

checks = [
    r"- \[ \] Mistral NeMo \(12B\)",
    r"- \[ \] Mistral Large / Pixtral \(structural proxy verification\)",
    r"- \[ \] DeepSeek Coder \(1\.3B, 6\.7B, 33B\)",
    r"- \[ \] DeepSeek Math",
    r"- \[ \] Llama 3\.1 \(8B, 70B, 400B\) - 128k context scaling parity",
    r"- \[ \] Llama 3\.2 \(1B, 3B, Vision variants\)",
    r"- \[ \] Gemma 1\.1 / RecurrentGemma",
    r"- \[ \] Phi-1 / Phi-1\.5 - Dense blocks",
    r"- \[ \] Phi-2",
    r"- \[ \] Phi-3\.5 \(MoE variants\)",
    r"- \[ \] Qwen 2 \(0\.5B to 72B\)",
    r"- \[ \] Qwen 2\.5 \(0\.5B to 72B\) - Dynamic vocabulary scaling",
    r"- \[ \] RoBERTa, DistilBERT, ALBERT, DeBERTa \(v1, v2, v3\)",
    r"- \[ \] FLAN-T5",
    r"- \[ \] GPT-J \(6B\), GPT-NeoX \(20B\)",
    r"- \[ \] xLSTM \(Vision and Text variants\)",
    r"- \[ \] RetNet \(Retentive Networks\)",
    r"- \[ \] Gated Linear Attention \(GLA\) Models",
    r"- \[ \] S3D \(Separable 3D CNNs\)",
    r"- \[ \] TimeSformer",
    r"- \[ \] VideoMAE",
    r"- \[ \] GraphSAGE",
    r"- \[ \] Informer",
    r"- \[ \] FourCastNet \(Adaptive Fourier Neural Operator - AFNO blocks\)",
    r"- \[ \] VITS \(Variational Inference with adversarial learning\)",
]

for check in checks:
    content = re.sub(check, check.replace("- [ ]", "- [x]"), content)

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "w") as f:
    f.write(content)
