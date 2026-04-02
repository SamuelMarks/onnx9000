# The Definitive PyTorch, Keras & Universal Interop Roadmap (`onnx9000`)

This document serves as the absolute architectural master plan for the `onnx9000` Python ecosystem. The goal is to create a seamless "Framework Switcheroo": the ability to ingest any model (PyTorch, Keras, JAX), target any high-performance environment (C, C++, WASM, WebNN, MLIR), and regenerate idiomatic source code in any framework.

---

## Pillar 1: Ingestion Frontends (Source -> IR)

Exhaustive capture of model semantics from all major Python AI frameworks.

### Section 1.1: PyTorch Mastery

- [x] **Phase 1: Basic Tracing (Tensor-only):** Current state. Functional for simple graphs.
- [x] **Phase 2: Advanced Tracing (Collections):** Support for nested `dict`, `list`, and `tuple` inputs/outputs in `trace()`.
- [x] **Phase 3: TorchScript IR Parsing:** Implement a `script()` frontend that walks the TorchScript Graph/Node/Value IR to capture static control flow.
- [x] **Phase 4: FX Symbolic Tracing:** Direct ingestion of `torch.fx.GraphModule`, enabling AOT capture without execution.
- [x] **Phase 5: torch.export (AOTInductor):** Support for the PyTorch 2.x export path, capturing clean AtenIR.
- [x] **Phase 6: AtenIR Decompositions:** Map the 200+ `aten::` operators directly to IR, minimizing external dependencies.
- [x] **Phase 7: Metadata & Hierarchy:** Preserve `nn.Module` names (e.g., `self.backbone.layer1`) for high-fidelity code regeneration.
- [x] **Phase 8: Dynamic Shape Tracking:** Capture symbolic dimension variables (`B`, `S`, `L`) from `torch.export`.
- [x] **Phase 9: Buffer & Parameter Parity:** Distinguish between trainable weights and non-trainable persistent buffers (e.g., BatchNorm means).
- [x] **Phase 10: Custom TorchBind Objects:** Ingest `torch::CustomClass` instances and map them to C++ class stubs.

### Section 1.2: Keras & JAX Universality

- [x] **Phase 11: Keras 3 Functional Parser:** Non-executing graph walker for Keras 3 models, resolving node reuse.
- [x] **Phase 12: Keras 3 Subclass Tracing:** Sandbox execution of `call()` methods to capture the graph of subclassed models.
- [x] **Phase 13: Keras 2 H5 Legacy Bridge:** Native parsing of legacy `.h5` files without requiring `tensorflow` installed.
- [x] **Phase 14: `keras.ops` Direct Mapping:** 1:1 mapping of the backend-agnostic Keras op-set to IR.
- [x] **Phase 15: JAX Pytree Resolution:** Flattening and un-flattening of JAX Pytree structures during IR conversion.
- [x] **Phase 16: JAX `jax.jit` Ingestion:** Capture the compiled HLO (High Level Optimizer) graph from JAX.
- [x] **Phase 17: Equinox/Flax Support:** Specific handling for state-in/state-out patterns common in JAX functional modules.

---

## Pillar 2: Domain-Specific Ops & Architectures

Deep support for specialized model families.

### Section 2.1: Computer Vision (CV)

- [x] **Phase 18: Spatial Ops Expansion:** Full support for `Conv1D/2D/3D` and `ConvTranspose`.
- [x] **Phase 19: Deformable Convolutions:** Implementation of offset-based convolution kernels in C++/WASM.
- [x] **Phase 20: ViT & Window Attention:** Optimized IR representations for Swin Transformer `Roll` and `Unroll` ops.

### Section 2.2: NLP & Generative AI

- [x] **Phase 21: FlashAttention Subgraphs:** Automatically lower standard Attention patterns into optimized Flash kernels.
- [x] **Phase 22: KV Cache Management:** First-class IR nodes for Key-Value cache updating in LLM inference.
- [x] **Phase 23: Rotary Embeddings (RoPE):** Optimized C++/WASM kernels for complex-number-like rotations.
- [x] **Phase 24: Text Preprocessing:** Support for `TextVectorization` and `Tokenizer` nodes within the core model graph.

### Section 2.3: Audio & Signal Processing

- [x] **Phase 25: STFT/ISTFT:** Bit-exact mapping of Short-Time Fourier Transform to IR.
- [x] **Phase 26: Mel-Filterbank Generation:** Implement Mel-scale conversion as a constant-folded initializer pass.
- [x] **Phase 27: DCT-II & MFCC:** Native support for Discrete Cosine Transform for speech features.

---

## Pillar 3: Advanced Graph Transformations

The "Middle-End" of the compiler pipeline.

### Section 3.1: Shape & Type Reasoning

- [x] **Phase 28: Symbolic Shape Solving:** Integration with `Z3` or `SymPy` to prove shape properties across the graph.
- [x] **Phase 29: Dynamic Shape Propagation:** Complete shape inference for nodes with data-dependent shapes (e.g., `NonMaxSuppression`).
- [x] **Phase 30: Type Promotion Engine:** Strict adherence to NumPy/PyTorch type promotion rules during IR construction.

### Section 3.2: Optimization & Fusion

- [x] **Phase 31: Horizontal Fusion:** Merging parallel operations (e.g., three `Gemm` calls into one) for wider memory bandwidth.
- [x] **Phase 32: Vertical Fusion:** Standard `Conv + BN + ReLU` fusion into monolithic loops.
- [x] **Phase 33: Layout Optimization:** Global optimization of `NCHW` vs `NHWC` to minimize unnecessary transposes.
- [x] **Phase 34: Constant Folding:** Aggressive pre-calculation of all subgraphs depending only on weights.
- [x] **Phase 35: Dead Code Elimination (DCE):** Recursive pruning of all unused output branches.

---

## Pillar 4: Numerical Precision & Hardware

Managing the bits and the hardware-specific intrinsics.

### Section 4.1: Quantization & Compression

- [x] **Phase 36: INT8/UINT8 PTQ:** Post-training quantization with calibration dataset support.
- [x] **Phase 37: INT8 QAT Ingestion:** Support for `FakeQuantize` nodes from PyTorch/Keras training.
- [x] **Phase 38: FP8 (E4M3/E5M2):** Native support for 8-bit floating point types for H100/L40S class hardware.
- [x] **Phase 39: 4-Bit AWQ/GPTQ:** Support for 4-bit packed weights and group-wise quantization.
- [x] **Phase 40: Weight Pruning:** Sparse matrix support (CSR/COO) for pruned model architectures.

### Section 4.2: Distributed Execution

- [x] **Phase 41: Pipeline Parallelism:** Automatic graph splitting for multi-device inference.
- [x] **Phase 42: Collective Ops Mapping:** Map `all_reduce`, `broadcast` to distributed C++ templates.
- [x] **Phase 43: Memory Sharding:** Support for sharded initializers across multiple RAM address spaces.

---

## Pillar 5: Target-Specific Code Generation

Excellence in the final emitted payload.

### Section 5.1: Embedded & Bare-Metal

- [x] **Phase 44: Strict C89 (onnx2c):** Complete coverage of 150+ operators in MISRA-C compliant code.
- [x] **Phase 45: Zero-Malloc Guarantee:** Static allocation of the entire memory arena at compile time.
- [x] **Phase 46: RTOS Task Wrappers:** Auto-generate FreeRTOS or Zephyr task skeletons for the model.
- [x] **Phase 47: CMSIS-NN/ESP-NN:** Target-specific intrinsics for ARM and Espressif microcontrollers.

### Section 5.2: Modern Native & Web

- [x] **Phase 48: C++ SIMD (AVX512/NEON):** Hand-rolled SIMD kernels for the native backend.
- [x] **Phase 49: WASM Threading:** Multi-threaded WASM inference via `SharedArrayBuffer` and Web Workers.
- [x] **Phase 50: WebNN / NPU Bridge:** Direct emission of WebNN graph construction code for hardware acceleration in the browser.
- [x] **Phase 51: WebGPU / WGSL:** Generate raw WGSL Compute Shaders for massive parallelization on the web.

### Section 5.3: Compiler Dialects

- [x] **Phase 52: MLIR TOSA Lowering:** Lower IR to the Tensor Operator Set Architecture dialect.
- [x] **Phase 53: MLIR Linalg Lowering:** Lower to linear algebra loops for affine transformations.
- [x] **Phase 54: StableHLO Support:** Export to the StableHLO dialect for OpenXLA compatibility.

---

## Pillar 6: The Great Reverse Pipeline (IR -> Source Code)

The ultimate feature: turning optimized graphs back into human-readable code.

### Section 6.1: Regeneration Logic

- [x] **Phase 55: PyTorch CodeGen:** Implement `generate_pytorch()` creating a full `nn.Module` class.
- [x] **Phase 56: Keras 3 CodeGen:** Implement `generate_keras()` using backend-agnostic `keras.ops`.
- [x] **Phase 57: JAX CodeGen:** Implement `generate_jax()` producing pure functions and Pytrees.
- [x] **Phase 58: Logic Reconstruction:** Detect sequences and group them into `nn.Sequential` or `keras.Sequential`.
- [x] **Phase 59: Layout Restoration:** Automatically insert transposes to return models to their native layout (`NCHW` for Torch, `NHWC` for Keras).

### Section 6.2: State & Weight Management

- [x] **Phase 60: State Dict Export:** Export model weights into `.pth` or `.bin` files that match the generated code.
- [x] **Phase 61: Universal Weight Bridge:** Standalone conversion between Safetensors, HDF5, and PyTorch pickles.
- [x] **Phase 62: Docstring Restoration:** Use IR metadata to restore original class and method documentation.

---

## Pillar 7: Ecosystem, Tooling & Enterprise

Validation and developer experience.

### Section 7.1: Tooling & CLI

- [x] **Phase 63: Universal Converter CLI:** `onnx9000 convert --from keras --to pytorch`.
- [x] **Phase 64: Numerical Debugger:** Step-by-step activation comparison between source and target.
- [x] **Phase 65: Memory Profiler:** Visual report of peak memory usage per layer on the target device.
- [x] **Phase 66: Integrated Graph Explorer:** Web-based IDE integration for visual debugging.

### Section 7.2: Validation & Enterprise

- [x] **Phase 67: Differential Fuzzing:** Automated tests that compare outputs across all 70 conversion paths.
- [x] **Phase 68: Model Zoo Parity:** Continuous regression testing for 50+ industry-standard architectures.
- [x] **Phase 69: Model Obfuscation:** Optional pass to rename tensors and nodes for intellectual property protection.
- [x] **Phase 70: Zero-Dependency Guarantee:** Strict CI enforcement that all generated code requires zero external libs.

---

## Current Maturity Matrix

| Path              | Status     | Target Maturity                                 |
| :---------------- | :--------- | :---------------------------------------------- |
| **PyTorch -> IR** | **STABLE** | Full coverage including Custom TorchBind.       |
| **Keras -> IR**   | **STABLE** | Keras 3 and Legacy H5 both stable.              |
| **JAX -> IR**     | **STABLE** | Jaxpr and Pytree resolution fully functional.   |
| **IR -> C (C89)** | **STABLE** | Highly mature for Embedded and Bare-Metal.      |
| **IR -> C++**     | **STABLE** | SIMD-optimized native backend active.           |
| **IR -> WASM**    | **STABLE** | Multi-threaded SIMD functional in browser.      |
| **IR -> MLIR**    | **STABLE** | TOSA, Linalg, and StableHLO dialects supported. |
| **IR -> Keras**   | **STABLE** | Full CodeGen for Keras 3.                       |
| **IR -> PyTorch** | **STABLE** | Full CodeGen for nn.Module.                     |
| **IR -> JAX**     | **STABLE** | Full CodeGen for Pytrees.                       |
