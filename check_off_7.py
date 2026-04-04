import re

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "r") as f:
    content = f.read()

checks = [
    r"- \[ \] DeepSeek V3 \(MoE with auxiliary loss structures\)",
    r"- \[ \] Qwen 1\.5 \(0\.5B to 110B\) - Dual-chunk attention mechanisms",
    r"- \[ \] Llama 3 \(8B, 70B\) - High-freq RoPE scaling, large vocab",
    r"- \[ \] Jamba - Hybrid Mamba \+ Transformer MoE",
    r"- \[ \] DenseNet \(121, 169, 201\)",
    r"- \[ \] DeepLabV3, DeepLabV3\+",
    r"- \[ \] Swin Transformer V2",
    r"- \[ \] Stable Diffusion 3\.0 \(SD3\) - MMDiT architecture parity",
    r"- \[ \] Flux\.1 \(schnell, dev\) - Rectified Flow matching, RoPE in vision",
    r"- \[ \] PointNet\+\+ \(Set Abstraction layers, FPS sampling logic\)",
]

for check in checks:
    content = re.sub(check, check.replace("- [ ]", "- [x]"), content)

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "w") as f:
    f.write(content)
