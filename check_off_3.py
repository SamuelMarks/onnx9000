import re

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "r") as f:
    content = f.read()

checks = [
    r"- \[ \] BERT \(Base, Large\)",
    r"- \[ \] GPT-2 \(Small, Medium, Large, XL\)",
    r"- \[ \] T5 \(Small, Base, Large, 3B, 11B\)",
    r"- \[ \] VGG \(11, 13, 16, 19, \+BatchNorm variants\)",
    r"- \[ \] Inception \(v1, v3, v4\), Inception-ResNet",
]

for check in checks:
    content = re.sub(check, check.replace("- [ ]", "- [x]"), content)

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "w") as f:
    f.write(content)
