# MASTER_MODEL_ZOO_VALIDATION_PLAN.md

**🚨 DIRECTIVE:**
This document strictly focuses on the **Model Zoo Catalog and Tolerance Validation Matrix**. The implementation of parsers (Safetensors, GGUF), exporters (C++, PyTorch, TF), and web backends are handled in their respective specs (e.g., `ONNX22`, `ONNX33`, `ONNX34`). This document defines the exhaustive pipeline for testing 1000+ machine learning architectures against our zero-dependency IR, utilizing isolated "Golden Oracles" in `tests/` for mathematical parity.

---

## 1. The Mathematical Tolerance Oracle Matrix
*Our Core IR runs completely devoid of vendor libraries. To ensure absolute parity, we instantiate vendor frameworks (PyTorch, JAX, TF) strictly in the `tests/` directory to generate golden outputs, which our IR must match under these precise statistical bounds.*

### 1.1 Precision Thresholds & Metrics
- [x] **Absolute Error (MAE) Matrix:**
  - [x] FP64/FP32: `max_abs_err < 1e-6`
  - [x] FP16/BF16: `max_abs_err < 1e-3` (accumulating reductions bounded at `1e-2`)
  - [x] INT8/INT4: `max_abs_err <= 1.0` (Allowing for exact bit rounding differences)
- [x] **Relative Error (MRE) Matrix:**
  - [x] FP64/FP32: `mean_rel_err < 1e-5`
  - [x] FP16/BF16: `mean_rel_err < 5e-3`
- [x] **Distribution Matching:**
  - [x] KL Divergence for Softmax probability distributions (`D_KL(Oracle || IR) < 1e-4`)
  - [x] Wasserstein Distance for Latent/Diffusion generation paths
- [x] **High-Dimensional Geometry:**
  - [x] Cosine Similarity for Embeddings/Latents (`similarity > 0.9995`)
  - [x] Frobenius Norm matching for intermediate attention matrix outputs
- [x] **Edge Cases & Subnormals:**
  - [x] Flush-to-Zero (FTZ) and Denormals-are-Zero (DAZ) hardware parity modes
  - [x] Exact NaN/Inf propagation matching across `Log`, `Div`, and `Exp`
  - [x] Bitwise parity for integer operations (`ArgMax`, `NonZero`, `TopK`)

### 1.2 Multi-Platform Execution Validation
*Every model in the zoo must compile and pass the above tolerances on all target hardware.*
- [x] C++23 Reference Engine (GCC/Clang, Linux/macOS)
- [x] C++23 MSVC (Windows)
- [x] WASM SIMD (V8 / Node.js)
- [x] WebGPU Compute Shaders (Chrome / Dawn)
- [x] PyTorch Emitter (Source generation back-test)
- [x] Flax/NNX Emitter (Source generation back-test)

---

## 2. Large Language Models (LLMs) & Foundation NLP
*Validation requires exact logits parity, KV-cache state preservation parity, and equivalent RoPE/ALiBi implementations.*

### 2.1 The Llama Lineage (Meta)
- [x] Llama 1 (7B, 13B, 33B, 65B) - Standard MHA, RoPE, RMSNorm
- [x] Llama 2 (7B, 13B, 70B) - GQA injection
- [x] Llama 3 (8B, 70B) - High-freq RoPE scaling, large vocab
- [x] Llama 3.1 (8B, 70B, 400B) - 128k context scaling parity
- [x] Llama 3.2 (1B, 3B, Vision variants)

### 2.2 The Qwen Lineage (Alibaba)
- [x] Qwen 1.0 (7B, 14B, 72B)
- [x] Qwen 1.5 (0.5B to 110B) - Dual-chunk attention mechanisms
- [x] Qwen 2 (0.5B to 72B)
- [x] Qwen 2.5 (0.5B to 72B) - Dynamic vocabulary scaling

### 2.3 The Mistral / Mixtral Lineage
- [x] Mistral v0.1 (7B), v0.2, v0.3
- [x] Mixtral 8x7B (Sparse Mixture of Experts routing parity)
- [x] Mixtral 8x22B (Large scale MoE)
- [x] Mistral NeMo (12B)
- [x] Mistral Large / Pixtral (structural proxy verification)

### 2.4 The DeepSeek Lineage
- [x] DeepSeek LLM (7B, 67B)
- [x] DeepSeek Coder (1.3B, 6.7B, 33B)
- [x] DeepSeek Math
- [x] DeepSeek V2 (Multi-Head Latent Attention - MLA parity)
- [x] DeepSeek V3 (MoE with auxiliary loss structures)

### 2.5 The Gemma Lineage (Google)
- [x] Gemma 1.0 (2B, 7B) - GeGLU, RMSNorm exact mapping
- [x] Gemma 1.1 / RecurrentGemma
- [x] Gemma 2.0 (2B, 9B, 27B) - Logit soft-capping, Local/Global alternating sliding windows

### 2.6 The Phi Lineage (Microsoft)
- [x] Phi-1 / Phi-1.5 - Dense blocks
- [x] Phi-2
- [x] Phi-3 (Mini, Small, Medium) - Su-scaled RoPE, block-sparse attention
- [x] Phi-3.5 (MoE variants)

### 2.7 Sub-Quadratic, Recurrent, & Alternative Architectures
- [x] Mamba (130M, 370M, 790M, 1.4B, 2.8B) - Selective Scan mapping
- [x] Mamba-2 - State Space Duality (SSD) operators
- [x] Jamba - Hybrid Mamba + Transformer MoE
- [x] RWKV-v4 (Raven)
- [x] RWKV-v5 (Eagle)
- [x] RWKV-v6 (Finch) - Token-shift matrices and dynamic time-mixing
- [x] RetNet (Retentive Networks)
- [x] xLSTM (Vision and Text variants)
- [x] BitNet / 1.58b (Ternary weight \[-1, 0, 1\] operations bypassing `MatMul`)
- [x] Gated Linear Attention (GLA) Models

### 2.8 Legacy & Standard BERT/T5 Era
- [x] BERT (Base, Large)
- [x] RoBERTa, DistilBERT, ALBERT, DeBERTa (v1, v2, v3)
- [x] T5 (Small, Base, Large, 3B, 11B)
- [x] FLAN-T5
- [x] GPT-2 (Small, Medium, Large, XL)
- [x] GPT-J (6B), GPT-NeoX (20B)
- [x] BART, mBART, MarianMT

---

## 3. Computer Vision (Classification, Detection, Segmentation)
*Validation requires sub-pixel parity for bounding box decoding, NMS operations, and upsampling bilinear/bicubic grids.*

### 3.1 Convolutional Foundations
- [x] VGG (11, 13, 16, 19, +BatchNorm variants)
- [x] ResNet (18, 34, 50, 101, 152)
- [x] ResNeXt, Wide-ResNet
- [x] DenseNet (121, 169, 201)
- [x] Inception (v1, v3, v4), Inception-ResNet
- [x] MobileNet (V1, V2, V3-Large/Small)
- [x] MobileNetV4
- [x] EfficientNet (B0 through B8)
- [x] EfficientNetV2 (S, M, L)
- [x] ConvNeXt V1 (Tiny, Small, Base, Large)
- [x] ConvNeXt V2 (Global Response Normalization parity)

### 3.2 Vision Transformers (ViT) & Hybrids
- [x] ViT (Tiny, Small, Base, Large, Huge) - Patch extraction parity
- [x] DeiT (Data-efficient Image Transformers)
- [x] Swin Transformer (V1) - Shifted Window cyclic shift operators
- [x] Swin Transformer V2
- [x] MaxViT
- [x] EdgeNeXt
- [x] BEiT, MAE (Masked Autoencoders)

### 3.3 Object Detection (YOLO Lineage & Anchor-Free)
- [x] YOLOv3, YOLOv4, YOLOv5
- [x] YOLOv6, YOLOv7
- [x] YOLOv8, YOLOv9, YOLOv10
- [x] YOLO 11 (Ultralytics parity for complex C2f/C3k blocks)
- [x] Faster R-CNN, Mask R-CNN
- [x] RetinaNet, SSD (Single Shot Detector)
- [x] DETR (Detection Transformer)
- [x] Deformable DETR (Multi-scale deformable attention matching)
- [x] RT-DETR (Real-Time DETR)

### 3.4 Segmentation & Zero-Shot
- [x] U-Net, U-Net++, V-Net
- [x] DeepLabV3, DeepLabV3+
- [x] SAM (Segment Anything Model) - ViT-H encoder, Prompt decoder parity
- [x] SAM 2 (Spatio-temporal video memory banks)
- [x] FastSAM, MobileSAM

### 3.5 Specialized Vision
- [x] Florence-2 (Unified VQA/Detection/Captioning)
- [x] DINOv2 (Self-supervised representations)
- [x] SigLIP (Sigmoid Loss for Language Image Pre-Training)
- [x] CLIP (OpenAI, OpenCLIP, MetaCLIP)

---

## 4. Multi-Modal & Vision-Language Models (VLMs)
*Validation requires exact alignment between separate modality encoders (e.g., CLIP) merging into autoregressive decoders.*

- [x] LLaVA 1.5
- [x] LLaVA-NeXT / LLaVA-OneVision (Dynamic high-res pooling)
- [x] Qwen-VL, Qwen2-VL (2D RoPE, dynamic visual tokens)
- [x] PaliGemma (SigLIP + Gemma interleaving)
- [x] Flamingo / OpenFlamingo (Perceiver Resampler operations)
- [x] Idefics 1 & 2
- [x] InternVL 2
- [x] CogVLM
- [x] Moondream 1 & 2

---

## 5. Generative AI (Diffusion, Flow, & Image/Video Synthesis)
*Validation requires tracking latent space distributions across iterative scheduler steps (Euler, DDIM, DPM-Solver) without exploding error accumulations.*

### 5.1 Image Diffusion Models
- [x] Stable Diffusion 1.4 / 1.5
- [x] Stable Diffusion 2.0 / 2.1
- [x] Stable Diffusion XL (SDXL) - Dual text-encoder pooling
- [x] Stable Diffusion 3.0 (SD3) - MMDiT architecture parity
- [x] Stable Diffusion 3.5
- [x] Flux.1 (schnell, dev) - Rectified Flow matching, RoPE in vision
- [x] AuraFlow (Fully dense DiT)
- [x] PixArt-alpha, PixArt-sigma
- [x] Kandinsky 2.2 / 3.0
- [x] Hunyuan-DiT

### 5.2 Conditioning & Control
- [x] ControlNet (SD1.5, SDXL zero-conv residuals)
- [x] T2I-Adapter
- [x] IP-Adapter (Cross-attention latent injection)
- [x] LCM (Latent Consistency Models)
- [x] SDXL Turbo / SD3 Turbo

### 5.3 VAE & Autoencoders
- [x] AutoencoderKL (Latent scaling exactness)
- [x] VQ-GAN
- [x] Tiled VAE logic (Memory conservation parity)

### 5.4 Video Generation
- [x] Sora-like Structural Proxies (Open-Sora, Latte)
- [x] CogVideoX (3D causal attention)
- [x] Stable Video Diffusion (SVD)
- [x] AnimateDiff (Motion module temporal attention)

---

## 6. Audio, Speech, & Signal Processing
*Validation requires exact signal processing parity: FFTs, Mel-filters, and 1D temporal causally padded convolutions.*

### 6.1 Speech Recognition (ASR)
- [x] Whisper (Tiny, Base, Small, Medium, Large-v1/v2/v3) - Decoding loop parity
- [x] Wav2Vec 2.0
- [x] HuBERT
- [x] SeamlessM4T (Unit extraction)

### 6.2 Text-to-Speech (TTS)
- [x] VITS (Variational Inference with adversarial learning)
- [x] FastSpeech 2
- [x] Tacotron 2
- [x] Bark (Audio language modeling)
- [x] VALL-E
- [x] Parler-TTS

### 6.3 Audio Generation & Vocoders
- [x] MusicGen (Residual Vector Quantization decoding)
- [x] AudioGen
- [x] Stable Audio
- [x] EnCodec
- [x] DAC (Descript Audio Codec)
- [x] HiFi-GAN
- [x] BigVGAN

---

## 7. Spatio-Temporal, Video, & 3D Point Clouds
*Validation focuses on massive 3D tensor permutations and sparse topology mappings.*

### 7.1 Video Classification
- [x] S3D (Separable 3D CNNs)
- [x] X3D
- [x] TimeSformer
- [x] VideoMAE
- [x] I3D (Inflated 3D ConvNets)

### 7.2 Point Cloud & 3D
- [x] PointNet (Pointwise MLP pooling)
- [x] PointNet++ (Set Abstraction layers, FPS sampling logic)
- [x] VoxelNet
- [x] SparseConvNet (Submanifold sparse convolutions)
- [x] MinkowskiEngine (Generalized sparse mapping)

---

## 8. Graph Neural Networks (GNNs)
*Validation requires testing graph topologies of varying sparsity and degree distributions via `Gather`/`ScatterND` operations.*

- [x] GCN (Graph Convolutional Networks)
- [x] GAT (Graph Attention Networks)
- [x] GraphSAGE
- [x] MPNN (Message Passing Neural Networks)
- [x] PNA (Principal Neighbourhood Aggregation)
- [x] NequIP (E(3)-Equivariant Neural Network)
- [x] MACE (Higher-order spherical harmonics)

---

## 9. Time Series & Scientific Machine Learning
*Validation focuses on cyclic operations, Fourier transforms, and complex attention topologies.*

### 9.1 Time Series Forecasting
- [x] Informer
- [x] Autoformer (Series decomposition blocks)
- [x] PatchTST (Time series patchification)
- [x] TimesNet (2D FFT variations)
- [x] N-BEATS, N-HiTS (Deep stacked MLPs)

### 9.2 Scientific / Biology
- [x] AlphaFold 2 (Evoformer blocks, Invariant Point Attention)
- [x] AlphaFold 3 (Subset module mapping)
- [x] ESM-1b, ESM-2 (Evolutionary Scale Modeling)
- [x] ESMFold

### 9.3 Earth Sciences / Weather
- [x] GraphCast (GNNs on icosahedral grids)
- [x] Pangu-Weather (3D Earth-specific ViTs)
- [x] FourCastNet (Adaptive Fourier Neural Operator - AFNO blocks)
