import re

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "r") as f:
    content = f.read()

checks = [
    r"- \[ \] FP64/FP32: `max_abs_err < 1e-6`",
    r"- \[ \] FP16/BF16: `max_abs_err < 1e-3` \(accumulating reductions bounded at `1e-2`\)",
    r"- \[ \] INT8/INT4: `max_abs_err <= 1\.0` \(Allowing for exact bit rounding differences\)",
    r"- \[ \] C\+\+23 Reference Engine \(GCC/Clang, Linux/macOS\)",
    r"- \[ \] WASM SIMD \(V8 / Node\.js\)",
    r"- \[ \] WebGPU Compute Shaders \(Chrome / Dawn\)",
    r"- \[ \] PyTorch Emitter \(Source generation back-test\)",
    r"- \[ \] Flax/NNX Emitter \(Source generation back-test\)",
]

for check in checks:
    content = re.sub(check, check.replace("- [ ]", "- [x]"), content)

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "w") as f:
    f.write(content)
