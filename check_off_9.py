import re

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "r") as f:
    content = f.read()

checks = [
    r"- \[ \] DeiT \(Data-efficient Image Transformers\)",
    r"- \[ \] MaxViT",
    r"- \[ \] EdgeNeXt",
    r"- \[ \] YOLOv3, YOLOv4, YOLOv5",
    r"- \[ \] YOLOv6, YOLOv7",
    r"- \[ \] YOLOv8, YOLOv9, YOLOv10",
    r"- \[ \] RetinaNet, SSD \(Single Shot Detector\)",
    r"- \[ \] RT-DETR \(Real-Time DETR\)",
    r"- \[ \] SAM 2 \(Spatio-temporal video memory banks\)",
    r"- \[ \] FastSAM, MobileSAM",
    r"- \[ \] Florence-2 \(Unified VQA/Detection/Captioning\)",
    r"- \[ \] DINOv2 \(Self-supervised representations\)",
    r"- \[ \] LLaVA 1\.5",
    r"- \[ \] PaliGemma \(SigLIP \+ Gemma interleaving\)",
    r"- \[ \] Idefics 1 & 2",
    r"- \[ \] Tacotron 2",
    r"- \[ \] MinkowskiEngine \(Generalized sparse mapping\)",
    r"- \[ \] PNA \(Principal Neighbourhood Aggregation\)",
    r"- \[ \] NequIP \(E\(3\)-Equivariant Neural Network\)",
    r"- \[ \] AlphaFold 3 \(Subset module mapping\)",
]

for check in checks:
    content = re.sub(check, check.replace("- [ ]", "- [x]"), content)

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "w") as f:
    f.write(content)
