import re

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "r") as f:
    content = f.read()

checks = [
    r"- \[ \] Gemma 1\.0 \(2B, 7B\) - GeGLU, RMSNorm exact mapping",
    r"- \[ \] Mamba-2 - State Space Duality \(SSD\) operators",
    r"- \[ \] MobileNet \(V1, V2, V3-Large/Small\)",
    r"- \[ \] U-Net, U-Net\+\+, V-Net",
    r"- \[ \] Stable Diffusion 1\.4 / 1\.5",
    r"- \[ \] Wav2Vec 2\.0",
]

for check in checks:
    content = re.sub(check, check.replace("- [ ]", "- [x]"), content)

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "w") as f:
    f.write(content)
