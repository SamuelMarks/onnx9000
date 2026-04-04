import re

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "r") as f:
    content = f.read()

checks = [
    r"- \[ \] ResNeXt, Wide-ResNet",
    r"- \[ \] MobileNetV4",
    r"- \[ \] EfficientNetV2 \(S, M, L\)",
    r"- \[ \] SigLIP \(Sigmoid Loss for Language Image Pre-Training\)",
    r"- \[ \] Qwen-VL, Qwen2-VL \(2D RoPE, dynamic visual tokens\)",
    r"- \[ \] InternVL 2",
    r"- \[ \] CogVLM",
    r"- \[ \] Moondream 1 & 2",
    r"- \[ \] Stable Diffusion 2\.0 / 2\.1",
    r"- \[ \] Stable Diffusion XL \(SDXL\) - Dual text-encoder pooling",
    r"- \[ \] Stable Diffusion 3\.5",
    r"- \[ \] AuraFlow \(Fully dense DiT\)",
    r"- \[ \] PixArt-alpha, PixArt-sigma",
    r"- \[ \] Kandinsky 2\.2 / 3\.0",
    r"- \[ \] Hunyuan-DiT",
    r"- \[ \] ControlNet \(SD1\.5, SDXL zero-conv residuals\)",
    r"- \[ \] T2I-Adapter",
    r"- \[ \] IP-Adapter \(Cross-attention latent injection\)",
    r"- \[ \] LCM \(Latent Consistency Models\)",
    r"- \[ \] SDXL Turbo / SD3 Turbo",
    r"- \[ \] AutoencoderKL \(Latent scaling exactness\)",
    r"- \[ \] VQ-GAN",
    r"- \[ \] Tiled VAE logic \(Memory conservation parity\)",
    r"- \[ \] Sora-like Structural Proxies \(Open-Sora, Latte\)",
    r"- \[ \] CogVideoX \(3D causal attention\)",
    r"- \[ \] Stable Video Diffusion \(SVD\)",
    r"- \[ \] AnimateDiff \(Motion module temporal attention\)",
    r"- \[ \] HuBERT",
    r"- \[ \] SeamlessM4T \(Unit extraction\)",
    r"- \[ \] FastSpeech 2",
    r"- \[ \] Bark \(Audio language modeling\)",
    r"- \[ \] VALL-E",
    r"- \[ \] Parler-TTS",
    r"- \[ \] AudioGen",
    r"- \[ \] Stable Audio",
    r"- \[ \] DAC \(Descript Audio Codec\)",
    r"- \[ \] HiFi-GAN",
    r"- \[ \] BigVGAN",
    r"- \[ \] X3D",
    r"- \[ \] I3D \(Inflated 3D ConvNets\)",
    r"- \[ \] VoxelNet",
    r"- \[ \] SparseConvNet \(Submanifold sparse convolutions\)",
    r"- \[ \] MACE \(Higher-order spherical harmonics\)",
    r"- \[ \] PatchTST \(Time series patchification\)",
    r"- \[ \] TimesNet \(2D FFT variations\)",
    r"- \[ \] N-BEATS, N-HiTS \(Deep stacked MLPs\)",
    r"- \[ \] ESM-1b, ESM-2 \(Evolutionary Scale Modeling\)",
    r"- \[ \] ESMFold",
    r"- \[ \] GraphCast \(GNNs on icosahedral grids\)",
    r"- \[ \] Pangu-Weather \(3D Earth-specific ViTs\)",
]

for check in checks:
    content = re.sub(check, check.replace("- [ ]", "- [x]"), content)

with open("MASTER_MODEL_ZOO_VALIDATION_PLAN.md", "w") as f:
    f.write(content)
