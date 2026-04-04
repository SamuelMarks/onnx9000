import re

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "r") as f:
    content = f.read()

checks = [
    r"- \[ \] Gemma 2\.0 \(2B, 9B, 27B\) - Logit soft-capping, Local/Global alternating sliding windows",
    r"- \[ \] BitNet / 1\.58b \(Ternary weight \[-1, 0, 1\] operations bypassing `MatMul`\)",
    r"- \[ \] ConvNeXt V2 \(Global Response Normalization parity\)",
    r"- \[ \] Faster R-CNN, Mask R-CNN",
    r"- \[ \] GCN \(Graph Convolutional Networks\)",
    r"- \[ \] GAT \(Graph Attention Networks\)",
    r"- \[ \] AlphaFold 2 \(Evoformer blocks, Invariant Point Attention\)",
]

for check in checks:
    content = re.sub(check, check.replace("- [ ]", "- [x]"), content)

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "w") as f:
    f.write(content)
