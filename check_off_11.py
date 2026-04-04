import re

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "r") as f:
    content = f.read()

checks = [
    r"- \[ \] \*\*Absolute Error \(MAE\) Matrix:\*\*",
    r"- \[ \] \*\*Relative Error \(MRE\) Matrix:\*\*",
    r"- \[ \] FP64/FP32: `mean_rel_err < 1e-5`",
    r"- \[ \] FP16/BF16: `mean_rel_err < 5e-3`",
    r"- \[ \] \*\*Distribution Matching:\*\*",
    r"- \[ \] KL Divergence for Softmax probability distributions \(`D_KL\(Oracle \|\| IR\) < 1e-4`\)",
    r"- \[ \] Wasserstein Distance for Latent/Diffusion generation paths",
    r"- \[ \] \*\*High-Dimensional Geometry:\*\*",
    r"- \[ \] Cosine Similarity for Embeddings/Latents \(`similarity > 0\.9995`\)",
    r"- \[ \] Frobenius Norm matching for intermediate attention matrix outputs",
    r"- \[ \] \*\*Edge Cases & Subnormals:\*\*",
    r"- \[ \] Flush-to-Zero \(FTZ\) and Denormals-are-Zero \(DAZ\) hardware parity modes",
    r"- \[ \] Exact NaN/Inf propagation matching across `Log`, `Div`, and `Exp`",
    r"- \[ \] Bitwise parity for integer operations \(`ArgMax`, `NonZero`, `TopK`\)",
    r"- \[ \] C\+\+23 MSVC \(Windows\)",
    r"- \[ \] Mixtral 8x22B \(Large scale MoE\)",
    r"- \[ \] DeepSeek LLM \(7B, 67B\)",
    r"- \[ \] RWKV-v5 \(Eagle\)",
    r"- \[ \] BART, mBART, MarianMT",
]

for check in checks:
    content = re.sub(check, check.replace("- [ ]", "- [x]"), content)

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "w") as f:
    f.write(content)
