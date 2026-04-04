# MASTER_MODEL_ZOO_VALIDATION_PLAN.md

**🚨 DIRECTIVE:**
This document strictly focuses on the **Model Zoo Catalog and Tolerance Validation Matrix**. The implementation of parsers (Safetensors, GGUF), exporters (C++, PyTorch, TF), and web backends are handled in their respective specs (e.g., `ONNX22`, `ONNX33`, `ONNX34`). This document defines the exhaustive pipeline for testing 1000+ machine learning architectures against our zero-dependency IR, utilizing isolated "Golden Oracles" in `tests/` for mathematical parity.

---

## 1. The Mathematical Tolerance Oracle Matrix
*Our Core IR runs completely devoid of vendor libraries. To ensure absolute parity, we instantiate vendor frameworks (PyTorch, JAX, TF) strictly in the `tests/` directory to generate golden outputs, which our IR must match under these precise statistical bounds.*

### 1.1 Precision Thresholds & Metrics
- \[ \] \*\*Absolute Error \(MAE\) Matrix:\*\*
  - \[ \] FP64/FP32: `max_abs_err < 1e-6`
  - \[ \] FP16/BF16: `max_abs_err < 1e-3` \(accumulating reductions bounded at `1e-2`\)
  - \[ \] INT8/INT4: `max_abs_err <= 1\.0` \(Allowing for exact bit rounding differences\)
- \[ \] \*\*Relative Error \(MRE\) Matrix:\*\*
  - \[ \] FP64/FP32: `mean_rel_err < 1e-5`
  - \[ \] FP16/BF16: `mean_rel_err < 5e-3`
- \[ \] \*\*Distribution Matching:\*\*
  - \[ \] KL Divergence for Softmax probability distributions \(`D_KL\(Oracle \|\| IR\) < 1e-4`\)
  - \[ \] Wasserstein Distance for Latent/Diffusion generation paths
- \[ \] \*\*High-Dimensional Geometry:\*\*
  - \[ \] Cosine Similarity for Embeddings/Latents \(`similarity > 0\.9995`\)
  - \[ \] Frobenius Norm matching for intermediate attention matrix outputs
- \[ \] \*\*Edge Cases & Subnormals:\*\*
  - \[ \] Flush-to-Zero \(FTZ\) and Denormals-are-Zero \(DAZ\) hardware parity modes
  - \[ \] Exact NaN/Inf propagation matching across `Log`, `Div`, and `Exp`
  - \[ \] Bitwise parity for integer operations \(`ArgMax`, `NonZero`, `TopK`\)

### 1.2 Multi-Platform Execution Validation
*Every model in the zoo must compile and pass the above tolerances on all target hardware.*
- \[ \] C\+\+23 Reference Engine \(GCC/Clang, Linux/macOS\)
- \[ \] C\+\+23 MSVC \(Windows\)
- \[ \] WASM SIMD \(V8 / Node\.js\)
- \[ \] WebGPU Compute Shaders \(Chrome / Dawn\)
- \[ \] PyTorch Emitter \(Source generation back-test\)
- \[ \] Flax/NNX Emitter \(Source generation back-test\)

---

## 2. Large Language Models (LLMs) & Foundation NLP
*Validation requires exact logits parity, KV-cache state preservation parity, and equivalent RoPE/ALiBi implementations.*

### 2.1 The Llama Lineage (Meta)
- \[ \] Llama 1 \(7B, 13B, 33B, 65B\) - Standard MHA, RoPE, RMSNorm
- \[ \] Llama 2 \(7B, 13B, 70B\) - GQA injection
- \[ \] Llama 3 \(8B, 70B\) - High-freq RoPE scaling, large vocab
- \[ \] Llama 3\.1 \(8B, 70B, 400B\) - 128k context scaling parity
- \[ \] Llama 3\.2 \(1B, 3B, Vision variants\)

### 2.2 The Qwen Lineage (Alibaba)
- \[ \] Qwen 1\.0 \(7B, 14B, 72B\)
- \[ \] Qwen 1\.5 \(0\.5B to 110B\) - Dual-chunk attention mechanisms
- \[ \] Qwen 2 \(0\.5B to 72B\)
- \[ \] Qwen 2\.5 \(0\.5B to 72B\) - Dynamic vocabulary scaling

### 2.3 The Mistral / Mixtral Lineage
- \[ \] Mistral v0\.1 \(7B\), v0\.2, v0\.3
- \[ \] Mixtral 8x7B \(Sparse Mixture of Experts routing parity\)
- \[ \] Mixtral 8x22B \(Large scale MoE\)
- \[ \] Mistral NeMo \(12B\)
- \[ \] Mistral Large / Pixtral \(structural proxy verification\)

### 2.4 The DeepSeek Lineage
- \[ \] DeepSeek LLM \(7B, 67B\)
- \[ \] DeepSeek Coder \(1\.3B, 6\.7B, 33B\)
- \[ \] DeepSeek Math
- \[ \] DeepSeek V2 \(Multi-Head Latent Attention - MLA parity\)
- \[ \] DeepSeek V3 \(MoE with auxiliary loss structures\)

### 2.5 The Gemma Lineage (Google)
- \[ \] Gemma 1\.0 \(2B, 7B\) - GeGLU, RMSNorm exact mapping
- \[ \] Gemma 1\.1 / RecurrentGemma
- \[ \] Gemma 2\.0 \(2B, 9B, 27B\) - Logit soft-capping, Local/Global alternating sliding windows

### 2.6 The Phi Lineage (Microsoft)
- \[ \] Phi-1 / Phi-1\.5 - Dense blocks
- \[ \] Phi-2
- \[ \] Phi-3 \(Mini, Small, Medium\) - Su-scaled RoPE, block-sparse attention
- \[ \] Phi-3\.5 \(MoE variants\)

### 2.7 Sub-Quadratic, Recurrent, & Alternative Architectures
- \[ \] Mamba \(130M, 370M, 790M, 1\.4B, 2\.8B\) - Selective Scan mapping
- \[ \] Mamba-2 - State Space Duality \(SSD\) operators
- \[ \] Jamba - Hybrid Mamba \+ Transformer MoE
- \[ \] RWKV-v4 \(Raven\)
- \[ \] RWKV-v5 \(Eagle\)
- \[ \] RWKV-v6 \(Finch\) - Token-shift matrices and dynamic time-mixing
- \[ \] RetNet \(Retentive Networks\)
- \[ \] xLSTM \(Vision and Text variants\)
- \[ \] BitNet / 1\.58b \(Ternary weight \[-1, 0, 1\] operations bypassing `MatMul`\)
- \[ \] Gated Linear Attention \(GLA\) Models

### 2.8 Legacy & Standard BERT/T5 Era
- \[ \] BERT \(Base, Large\)
- \[ \] RoBERTa, DistilBERT, ALBERT, DeBERTa \(v1, v2, v3\)
- \[ \] T5 \(Small, Base, Large, 3B, 11B\)
- \[ \] FLAN-T5
- \[ \] GPT-2 \(Small, Medium, Large, XL\)
- \[ \] GPT-J \(6B\), GPT-NeoX \(20B\)
- \[ \] BART, mBART, MarianMT

---

## 3. Computer Vision (Classification, Detection, Segmentation)
*Validation requires sub-pixel parity for bounding box decoding, NMS operations, and upsampling bilinear/bicubic grids.*

### 3.1 Convolutional Foundations
- \[ \] VGG \(11, 13, 16, 19, \+BatchNorm variants\)
- \[ \] ResNet \(18, 34, 50, 101, 152\)
- \[ \] ResNeXt, Wide-ResNet
- \[ \] DenseNet \(121, 169, 201\)
- \[ \] Inception \(v1, v3, v4\), Inception-ResNet
- \[ \] MobileNet \(V1, V2, V3-Large/Small\)
- \[ \] MobileNetV4
- \[ \] EfficientNet \(B0 through B8\)
- \[ \] EfficientNetV2 \(S, M, L\)
- \[ \] ConvNeXt V1 \(Tiny, Small, Base, Large\)
- \[ \] ConvNeXt V2 \(Global Response Normalization parity\)

### 3.2 Vision Transformers (ViT) & Hybrids
- \[ \] ViT \(Tiny, Small, Base, Large, Huge\) - Patch extraction parity
- \[ \] DeiT \(Data-efficient Image Transformers\)
- \[ \] Swin Transformer \(V1\) - Shifted Window cyclic shift operators
- \[ \] Swin Transformer V2
- \[ \] MaxViT
- \[ \] EdgeNeXt
- \[ \] BEiT, MAE \(Masked Autoencoders\)

### 3.3 Object Detection (YOLO Lineage & Anchor-Free)
- \[ \] YOLOv3, YOLOv4, YOLOv5
- \[ \] YOLOv6, YOLOv7
- \[ \] YOLOv8, YOLOv9, YOLOv10
- \[ \] YOLO 11 \(Ultralytics parity for complex C2f/C3k blocks\)
- \[ \] Faster R-CNN, Mask R-CNN
- \[ \] RetinaNet, SSD \(Single Shot Detector\)
- \[ \] DETR \(Detection Transformer\)
- \[ \] Deformable DETR \(Multi-scale deformable attention matching\)
- \[ \] RT-DETR \(Real-Time DETR\)

### 3.4 Segmentation & Zero-Shot
- \[ \] U-Net, U-Net\+\+, V-Net
- \[ \] DeepLabV3, DeepLabV3\+
- \[ \] SAM \(Segment Anything Model\) - ViT-H encoder, Prompt decoder parity
- \[ \] SAM 2 \(Spatio-temporal video memory banks\)
- \[ \] FastSAM, MobileSAM

### 3.5 Specialized Vision
- \[ \] Florence-2 \(Unified VQA/Detection/Captioning\)
- \[ \] DINOv2 \(Self-supervised representations\)
- \[ \] SigLIP \(Sigmoid Loss for Language Image Pre-Training\)
- \[ \] CLIP \(OpenAI, OpenCLIP, MetaCLIP\)

---

## 4. Multi-Modal & Vision-Language Models (VLMs)
*Validation requires exact alignment between separate modality encoders (e.g., CLIP) merging into autoregressive decoders.*

- \[ \] LLaVA 1\.5
- \[ \] LLaVA-NeXT / LLaVA-OneVision \(Dynamic high-res pooling\)
- \[ \] Qwen-VL, Qwen2-VL \(2D RoPE, dynamic visual tokens\)
- \[ \] PaliGemma \(SigLIP \+ Gemma interleaving\)
- \[ \] Flamingo / OpenFlamingo \(Perceiver Resampler operations\)
- \[ \] Idefics 1 & 2
- \[ \] InternVL 2
- \[ \] CogVLM
- \[ \] Moondream 1 & 2

---

## 5. Generative AI (Diffusion, Flow, & Image/Video Synthesis)
*Validation requires tracking latent space distributions across iterative scheduler steps (Euler, DDIM, DPM-Solver) without exploding error accumulations.*

### 5.1 Image Diffusion Models
- \[ \] Stable Diffusion 1\.4 / 1\.5
- \[ \] Stable Diffusion 2\.0 / 2\.1
- \[ \] Stable Diffusion XL \(SDXL\) - Dual text-encoder pooling
- \[ \] Stable Diffusion 3\.0 \(SD3\) - MMDiT architecture parity
- \[ \] Stable Diffusion 3\.5
- \[ \] Flux\.1 \(schnell, dev\) - Rectified Flow matching, RoPE in vision
- \[ \] AuraFlow \(Fully dense DiT\)
- \[ \] PixArt-alpha, PixArt-sigma
- \[ \] Kandinsky 2\.2 / 3\.0
- \[ \] Hunyuan-DiT

### 5.2 Conditioning & Control
- \[ \] ControlNet \(SD1\.5, SDXL zero-conv residuals\)
- \[ \] T2I-Adapter
- \[ \] IP-Adapter \(Cross-attention latent injection\)
- \[ \] LCM \(Latent Consistency Models\)
- \[ \] SDXL Turbo / SD3 Turbo

### 5.3 VAE & Autoencoders
- \[ \] AutoencoderKL \(Latent scaling exactness\)
- \[ \] VQ-GAN
- \[ \] Tiled VAE logic \(Memory conservation parity\)

### 5.4 Video Generation
- \[ \] Sora-like Structural Proxies \(Open-Sora, Latte\)
- \[ \] CogVideoX \(3D causal attention\)
- \[ \] Stable Video Diffusion \(SVD\)
- \[ \] AnimateDiff \(Motion module temporal attention\)

---

## 6. Audio, Speech, & Signal Processing
*Validation requires exact signal processing parity: FFTs, Mel-filters, and 1D temporal causally padded convolutions.*

### 6.1 Speech Recognition (ASR)
- \[ \] Whisper \(Tiny, Base, Small, Medium, Large-v1/v2/v3\) - Decoding loop parity
- \[ \] Wav2Vec 2\.0
- \[ \] HuBERT
- \[ \] SeamlessM4T \(Unit extraction\)

### 6.2 Text-to-Speech (TTS)
- \[ \] VITS \(Variational Inference with adversarial learning\)
- \[ \] FastSpeech 2
- \[ \] Tacotron 2
- \[ \] Bark \(Audio language modeling\)
- \[ \] VALL-E
- \[ \] Parler-TTS

### 6.3 Audio Generation & Vocoders
- \[ \] MusicGen \(Residual Vector Quantization decoding\)
- \[ \] AudioGen
- \[ \] Stable Audio
- \[ \] EnCodec
- \[ \] DAC \(Descript Audio Codec\)
- \[ \] HiFi-GAN
- \[ \] BigVGAN

---

## 7. Spatio-Temporal, Video, & 3D Point Clouds
*Validation focuses on massive 3D tensor permutations and sparse topology mappings.*

### 7.1 Video Classification
- \[ \] S3D \(Separable 3D CNNs\)
- \[ \] X3D
- \[ \] TimeSformer
- \[ \] VideoMAE
- \[ \] I3D \(Inflated 3D ConvNets\)

### 7.2 Point Cloud & 3D
- \[ \] PointNet \(Pointwise MLP pooling\)
- \[ \] PointNet\+\+ \(Set Abstraction layers, FPS sampling logic\)
- \[ \] VoxelNet
- \[ \] SparseConvNet \(Submanifold sparse convolutions\)
- \[ \] MinkowskiEngine \(Generalized sparse mapping\)

---

## 8. Graph Neural Networks (GNNs)
*Validation requires testing graph topologies of varying sparsity and degree distributions via `Gather`/`ScatterND` operations.*

- \[ \] GCN \(Graph Convolutional Networks\)
- \[ \] GAT \(Graph Attention Networks\)
- \[ \] GraphSAGE
- \[ \] MPNN \(Message Passing Neural Networks\)
- \[ \] PNA \(Principal Neighbourhood Aggregation\)
- \[ \] NequIP \(E\(3\)-Equivariant Neural Network\)
- \[ \] MACE \(Higher-order spherical harmonics\)

---

## 9. Time Series & Scientific Machine Learning
*Validation focuses on cyclic operations, Fourier transforms, and complex attention topologies.*

### 9.1 Time Series Forecasting
- \[ \] Informer
- \[ \] Autoformer \(Series decomposition blocks\)
- \[ \] PatchTST \(Time series patchification\)
- \[ \] TimesNet \(2D FFT variations\)
- \[ \] N-BEATS, N-HiTS \(Deep stacked MLPs\)

### 9.2 Scientific / Biology
- \[ \] AlphaFold 2 \(Evoformer blocks, Invariant Point Attention\)
- \[ \] AlphaFold 3 \(Subset module mapping\)
- \[ \] ESM-1b, ESM-2 \(Evolutionary Scale Modeling\)
- \[ \] ESMFold

### 9.3 Earth Sciences / Weather
- \[ \] GraphCast \(GNNs on icosahedral grids\)
- \[ \] Pangu-Weather \(3D Earth-specific ViTs\)
- \[ \] FourCastNet \(Adaptive Fourier Neural Operator - AFNO blocks\)
