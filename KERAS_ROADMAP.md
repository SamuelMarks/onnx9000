# The Ultimate Exhaustive Keras Integration Roadmap (`onnx9000`)

This document serves as the definitive, master architectural roadmap for fully integrating the Keras ecosystem (Keras 2.x and Keras 3.x Multi-Backend) into the `onnx9000` toolchain.

The goal is to achieve 100% mathematical and topological parity for **any arbitrary Keras model**, facilitating seamless translation to ONNX, aggressive compilation to Edge targets (Bare-metal C/C++, WASM SIMD, WebGPU, WebNN, MLIR), and perfect source-code regeneration back to Keras.

---

## Phase 1: Advanced Parsing & Topology Ingestion (Keras -> AST)

Exhaustive parsing of all Keras serialization formats, managing topological complexities across all major Keras versions.

- [x] Extract basic Keras topologies from TF.js `model.json`.
- [x] Parse structural configurations from Keras 2 HDF5 (`.h5`) via pure JS/TS `jsfive`.
- [x] Parse Keras 3 (`.keras` v3 archive format) utilizing JS zip extraction to read `metadata.json`, `config.json`.
- [x] Extract weights from Keras 3 `model.weights.h5` inside the `.keras` archive.
- [x] Extract Keras 3 `safetensors` weight backend (if PyTorch/JAX backend was used to save the `.keras` file).
- [x] Parse TF `SavedModel` format (extracting from `saved_model.pb` and `variables/variables.index` directly in JS).
- [x] Implement robust `Functional` API resolution, properly tracking shared layer instances (node reuse) across multiple inbound/outbound paths.
- [x] Implement robust `Sequential` API resolution.
- [x] Resolve Keras `Model` subclassing topologies by falling back to execution-trace extraction via Pyodide/Python `tf.autograph` tracing if static AST is unavailable.
- [x] Handle nested `Model` within `Model` configurations (sub-graph flattening).
- [x] Reassemble TF.js weight shards (`.bin` files) into unified Float32/Float16 contiguous memory buffers based on `weightsManifest`.
- [x] Build a Custom Layer Plugin API, allowing developers to register unhandled Keras layers mapping directly to ONNX subgraph injections.
- [x] Extract and map Keras model signatures (explicit input/output dictionary names, e.g., `serving_default`).
- [x] Parse `keras.constraints` (MaxNorm, NonNeg, UnitNorm) and embed as ONNX graph metadata for downstream QAT/compilation awareness.

## Phase 2: The Core API & Dense Layers (Keras -> ONNX)

- [x] **Core:** `Dense` -> ONNX `Gemm` or `MatMul` + `Add`.
- [x] **Core:** `Activation` -> Map to corresponding ONNX math ops.
- [x] **Core:** `Flatten` -> ONNX `Flatten` or `Reshape`.
- [x] **Core:** Map `Dropout`, `SpatialDropout1D/2D/3D`, `AlphaDropout` to ONNX `Identity` (Inference mode bypass).
- [x] **Core Expanded:** Translate `RepeatVector` to ONNX `Tile` or `Expand`.
- [x] **Core Expanded:** Translate `Masking` and `ComputeMasking` to explicit boolean tensor flows.
- [x] **Core Expanded:** Translate `Lambda` layers to ONNX subgraphs via dynamic symbolic execution / ONNX Script.
- [x] **Core Expanded:** `ActivityRegularization` -> ONNX `Identity` (Inference mode bypass).
- [x] **Core Expanded:** `EinsumDense` -> ONNX `Einsum` + `Add`.

## Phase 3: Convolution, Pooling & Vision Operations

- [x] **Convolution:** `Conv2D`, `SeparableConv2D` (partial).
- [x] **Convolution 1D/3D:** `Conv1D`, `Conv3D`, `SeparableConv1D`.
- [x] **Convolution Transpose:** `Conv2DTranspose`, `Conv3DTranspose`, `Conv1DTranspose`.
- [x] **Depthwise:** `DepthwiseConv2D`, `DepthwiseConv1D` (Mapped to ONNX `Conv` with `group == in_channels`).
- [x] **Advanced Conv:** Support Keras `groups` attribute correctly for group convolutions.
- [x] **Advanced Conv:** Support Keras `dilation_rate` mapping to ONNX `dilations`.
- [x] **Pooling:** `MaxPooling2D`, `AveragePooling2D`, `GlobalAveragePooling2D`, `GlobalMaxPooling2D`.
- [x] **Pooling 1D/3D:** `MaxPooling1D/3D`, `AveragePooling1D/3D`, `GlobalAveragePooling1D/3D`.
- [x] **UpSampling:** `UpSampling1D/2D/3D` (Map to ONNX `Resize` with `nearest` or `linear/bilinear` modes).
- [x] **Padding & Cropping:** `ZeroPadding1D/2D/3D`, `Cropping1D/2D/3D` (Map to ONNX `Pad` and `Slice`).
- [x] **Padding Modes:** Support Keras padding types: `"valid"`, `"same"`, `"causal"`.

## Phase 4: Recurrent Neural Networks (RNNs) & Stateful Graphs

- [x] **RNN Base:** `SimpleRNN` -> ONNX `RNN`.
- [x] **LSTM:** `LSTM` -> ONNX `LSTM`.
- [x] **GRU:** `GRU` -> ONNX `GRU`.
- [x] **RNN Configs:** Handle `return_sequences=True/False`.
- [x] **RNN Configs:** Handle `return_state=True/False` (outputting hidden/cell states).
- [x] **RNN Configs:** Handle `go_backwards=True` (Reverse sequence before feeding to ONNX RNN).
- [x] **RNN Configs:** Handle `time_major=True/False` (Inject ONNX `Transpose` to swap Batch and Sequence dims).
- [x] **RNN Configs:** Handle `unroll=True` (Statically unrolling the RNN cell into standard ONNX Math nodes).
- [x] **RNN Cells:** Support explicitly defined `SimpleRNNCell`, `LSTMCell`, `GRUCell`, `StackedRNNCells`.
- [x] **RNN Wrappers:** Support `Bidirectional` wrapper (Mapped to `direction="bidirectional"` in ONNX).
- [x] **RNN Wrappers:** Support `TimeDistributed` wrapper (Mapped via ONNX `Reshape` -> Op -> `Reshape` pattern).
- [x] **ConvRNN:** `ConvLSTM1D/2D/3D` -> Unrolled ONNX subgraphs combining `Conv` and `LSTM` logic.

## Phase 5: Attention, Transformers & NLP Preprocessing

- [x] **Attention:** Translate `MultiHeadAttention` down to standard ONNX `MatMul`, `Transpose`, `Softmax` subgraphs.
- [x] **Attention:** Support `Attention` (Luong) and `AdditiveAttention` (Bahdanau).
- [x] **Attention Masks:** Propagate Keras `attention_mask` securely into ONNX `Where` + `Add (-1e9)` logic.
- [x] **Embeddings:** `Embedding` -> ONNX `Gather`.
- [x] **Text/NLP:** Translate `TextVectorization` into ONNX string/hash mapping topologies (if ONNX `ai.onnx.ml` string operators are enabled).
- [x] **Text/NLP:** `StringLookup`, `IntegerLookup`, `Hashing` -> ONNX `CategoryMapper`.
- [x] **Tokenizers:** Support KerasNLP Tokenizers (WordPiece, SentencePiece, BPE) transpilation to ONNX `StringNormalizer` + `Tokenizer`.

## Phase 6: Audio, Speech & Signal Processing

- [x] **Spectrograms:** Translate Keras Audio `MelSpectrogram` into ONNX `STFT` + `MatMul` + `Abs`.
- [x] **MFCC:** Translate Keras `MFCC` to ONNX explicit DCT-II operations.
- [x] **STFT:** Support `tf.signal.stft` or Keras 3 equivalent mappings to ONNX `STFT`.

## Phase 7: Advanced Activations, Normalizations & Regularization

- [x] **Advanced Activations:** `LeakyReLU`, `PReLU`, `ELU`, `ThresholdedReLU`.
- [x] **Advanced Activations:** `Softmax` (with explicit `axis` translation), `Softplus`, `Softsign`.
- [x] **Modern Activations:** `Mish`, `GELU` (exact & approximate), `SELU`, `Swish`, `HardSwish`, `HardSigmoid`.
- [x] **Normalization:** `BatchNormalization` (Map `gamma`, `beta`, `moving_mean`, `moving_variance` to ONNX `BatchNormalization`).
- [x] **Normalization:** `LayerNormalization` -> ONNX `LayerNormalization` or `ReduceMean`/`ReduceVariance` subgraph.
- [x] **Normalization:** `UnitNormalization` -> ONNX `LpNormalization`.
- [x] **Normalization:** `GroupNormalization` -> ONNX `Reshape` + `InstanceNormalization` + `Reshape`.
- [x] **Noise:** `GaussianNoise`, `GaussianDropout` -> ONNX `RandomNormalLike` + `Add`/`Mul` (During training mode only; bypassed in inference).

## Phase 8: Reshaping, Merging, & Tensor Manipulation

- [x] **Merge:** `Concatenate` -> ONNX `Concat`.
- [x] **Merge:** `Add`, `Subtract`, `Multiply`, `Minimum`, `Maximum` -> ONNX element-wise equivalents.
- [x] **Merge:** `Average` -> ONNX `Mean` or `Sum` + `Div`.
- [x] **Merge:** `Dot` -> ONNX `MatMul` (handling axes specification explicitly).
- [x] **Reshape:** `Reshape` (Safely translating Keras `-1` inferences to ONNX `-1`).
- [x] **Permute:** `Permute` -> ONNX `Transpose`.
- [x] **Image Preprocessing:** `Rescaling`, `Resizing`, `CenterCrop` -> ONNX `Mul`, `Resize`, `Slice`.
- [x] **Data Augmentation:** Map `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomCrop` to `Identity` (Inference mode bypass).

## Phase 9: Data Types, Mixed Precision & Quantization

- [x] **Type Preservation:** Map Keras `mixed_float16` policies to ONNX `tensor(float16)` operators and `Cast` nodes.
- [x] **Complex Types:** Map `complex64`, `complex128` to dual-tensor (real, imag) ONNX graphs (since ONNX natively lacks deep complex support).
- [x] **Quantization-Aware Training (QAT):** Detect TFMOT (`tensorflow_model_optimization`) wrappers and map to ONNX `QuantizeLinear` / `DequantizeLinear`.
- [x] **QKeras:** Support importing QKeras `QDense`, `QConv2D`, `QActivation` strictly into ONNX `QLinearConv`, `QLinearMatMul`.
- [x] **Weight Packing:** Convert Keras 8-bit quantized weights natively into ONNX `uint8` / `int8` initializers.
- [x] **4-Bit AWQ/GPTQ:** Parse Keras 3 custom 4-bit packed weights and map to ONNX `MatMulNBits` or `DynamicQuantizeLinear`.

## Phase 10: Memory Layout & Dimension Resolution (NHWC to NCHW)

- [x] Fully implement `NHWC` -> `NCHW` automated transpositions for all spatial inputs (`Conv`, `Pool`, `UpSampling`).
- [x] Automate Keras Kernel Transpositions: e.g., Conv2D weights `[H, W, In, Out]` -> ONNX `[Out, In, H, W]`.
- [x] Automate Dense Kernel Transpositions: `[In, Out]` -> ONNX `[Out, In]` (if required by ONNX `Gemm`).
- [x] Automate Group Conv Transpositions: `[H, W, In/G, Out]` -> `[Out, In/G, H, W]`.
- [x] Dynamic Batch Resolution: Ensure Keras `None` batch dimensions strictly map to ONNX `-1` or named string parameters (`batch_size`).
- [x] Explicit Masking Propagation: Ensure `_keras_mask` tensor generation is natively translated into ONNX parallel boolean graphs.

## Phase 11: Bare-Metal C/C++ Generation (`onnx2c`)

- [x] Dynamically allocate static memory sizes and emit loops for Keras Dense/Conv outputs based on empirical shape inference.
- [x] Implement static C-struct generation for stateful Keras Recurrent layers (`LSTM`, `GRU`).
- [x] Implement native C math loop mappings for all Keras Advanced Activations (`GELU`, `Swish`, `Mish`).
- [x] Ensure Keras `DepthwiseConv2D` compiles to a specialized nested loop in C avoiding standard `Gemm` overhead.
- [x] Provide `#pragma GCC unroll` hints for Keras models with known static dimensions.
- [x] MISRA-C Compliance pass for generated C code from Keras topologies.
- [x] Inject CMSIS-NN ARM intrinsic headers specifically for Keras models converted from QAT (INT8).
- [x] Inject ESP-NN intrinsics for Espressif targets.
- [x] Provide `PROGMEM` macro generation for Keras weights going to Arduino targets.
- [x] Support generating an RTOS task-wrapper (FreeRTOS) for the Keras inference loop.

## Phase 12: WASM SIMD & WebAssembly Multithreading

- [x] Guarantee `MultiHeadAttention` mappings from Keras lower safely into WebAssembly `v128` instructions without memory fragmentation.
- [x] Ensure Keras `DepthwiseConv2D` hits highly optimized unrolled WASM SIMD `v128.load` / `f32x4.mul` loops.
- [x] Map Keras `Softmax` to a numerically stable WASM SIMD vector reduction loop.
- [x] Support multithreading Keras `BatchMatMul` (e.g., from Attention blocks) across multiple Web Workers using `SharedArrayBuffer`.
- [x] Enforce `__attribute__((aligned(16)))` on Keras memory arenas emitted to WASM to ensure SIMD compatibility.

## Phase 13: WebGPU Compute & WGSL Shader Generation

- [x] Map Keras `Conv2D` -> ONNX -> Directly to WGSL Compute Shaders utilizing workgroup shared memory.
- [x] Optimize WGSL matrix multiplication tiles (e.g., 16x16 or 32x32) based on the original Keras `Dense` layer dimensions.
- [x] Map Keras Activation layers to fused WGSL operations (combining `MatMul` + `BiasAdd` + `ReLU` in a single shader dispatch).
- [x] Handle NHWC -> NCHW transpose entirely within the GPU shader boundaries to avoid CPU-to-GPU memory stalls.
- [x] Map Keras `LayerNormalization` to a parallel WGSL reduction shader.

## Phase 14: WebNN API Integration & NPU Hardware Routing

- [x] Map Keras structural paradigms (like `Conv2D` + `BatchNormalization` + `ReLU`) directly to WebNN `MLGraphBuilder` fused operations.
- [x] Map Keras `SeparableConv2D` to WebNN `builder.conv2d` with `groups` configured appropriately.
- [x] Support WebNN async execution scheduling (`builder.build()`, `context.compute()`).
- [x] Implement graceful fallback: If an NPU (via WebNN) does not support a specific Keras layer (e.g., `Einsum`), fall back to WASM SIMD for that specific node while keeping the rest on the NPU.

## Phase 15: MLIR & StableHLO Lowering

- [x] Convert Keras dynamic control flow (`tf.cond` in Lambda layers) into `stablehlo.custom_call` or `scf.if` MLIR dialects.
- [x] Map Keras `tf.while_loop` (from custom RNNs) to MLIR `scf.while` loops.
- [x] Lower Keras Conv/Dense structures into MLIR `tosa` (Tensor Operator Set Architecture) dialect.
- [x] Lower Keras structures to `linalg` dialect for advanced affine loop transformations.
- [x] Export standard LLVM IR from the translated Keras -> MLIR graph.

## Phase 16: The Reverse Pipeline (ONNX -> Keras AST)

Round-trip capabilities: converting standard ONNX models into idiomatic, human-readable Keras Python source code.

- [x] Establish base `KerasGenerator` generating raw Python Keras implementation strings.
- [x] Map ONNX foundational ops (`Conv`, `MatMul`, `Relu`, `MaxPool`) to Keras equivalent `layers.*`.
- [x] Implement ONNX `NCHW` -> Keras `NHWC` layout reversion (inserting Keras `Permute` or altering configurations) to guarantee idiomatic code.
- [x] Translate ONNX `Gemm` into Keras `Dense` or `Dot` depending on dimensions and attributes.
- [x] Translate ONNX `BatchNormalization` back into Keras `layers.BatchNormalization` (de-fusing constants if needed).
- [x] Translate ONNX `Shape`, `Gather`, `Unsqueeze` manipulation graphs into Keras `layers.Lambda(lambda x: tf.shape(x)...)`.
- [x] Translate ONNX control flow (`If`, `Loop`) into Keras custom `Layer` subclass `call()` methods utilizing `tf.cond` and `tf.while_loop`.
- [x] Translate ONNX `RNN`, `LSTM`, `GRU` back to Keras recurrent layers, securely mapping hidden states.
- [x] Fallback unmappable ONNX nodes into raw `tf.raw_ops` calls within a `Lambda` layer to guarantee 100% graph preservation.

## Phase 17: Reverse Pipeline (Keras 3 Multi-Backend CodeGen)

- [x] Update `KerasGenerator` to support emitting Keras 3 `keras.ops` syntax instead of strict TensorFlow logic.
- [x] Translate ONNX math operators to backend-agnostic `keras.ops.matmul`, `keras.ops.exp`, `keras.ops.concatenate`.
- [x] Generate Keras 3 Subclassed Models (`class MyModel(keras.Model):`) for complex non-sequential ONNX graphs.
- [x] Provide AST to executable `.keras` or `.h5` generation (embedding ONNX initializers back into HDF5 structures using `jsfive`).
- [x] Export ONNX Constants directly to NumPy `.npy` or `.npz` files for easy loading by the generated Keras script via `model.load_weights()`.

## Phase 18: Optimization Passes & Fusions

Run these passes prior to final compilation to ensure parity with standard Keras execution speeds.

- [x] Fuse Keras `Dense` + `BatchNormalization`.
- [x] Fuse Keras `Conv2D` + `BatchNormalization`.
- [x] Fuse Keras `DepthwiseConv2D` + `BatchNormalization`.
- [x] Fuse Keras `Conv2D` + `Add` + `ReLU` (Standard ResNet block fusion).
- [x] Eliminate sequential `Reshape` -> `Reshape` redundancies caused by layout conversions.
- [x] Completely remove `Identity` layers created by skipped `Dropout` layers to tighten the execution arena.
- [x] Constant Folding: Pre-calculate static Keras `Lambda` or math operations offline.

## Phase 19: Tooling, CLI & IDE Integration

- [x] **CLI:** Ensure `npx onnx9000 convert --from keras --to c++ model.h5` operates losslessly in a single CLI pass.
- [x] **CLI:** Add command `npx onnx9000 inspect model.keras` to print terminal-based topological summaries.
- [x] **VSCode Extension:** Implement hovering over Keras code to preview the resulting ONNX memory footprint.
- [x] **VSCode Extension:** Provide Keras-to-ONNX visual graph comparisons directly in the editor.
- [x] **Netron Integration:** Support embedding the `onnx9000` Keras parser into a web-view to visually debug Keras -> ONNX translation anomalies.

## Phase 20: Comprehensive Validation, Zoo Parity & Benchmarking

- [x] **Unit Tests:** 100% coverage of individual Keras Layer-to-ONNX translation mappings (over 150+ layer variants).
- [x] **Zoo Parity (Vision):** Validate ResNet50, MobileNetV2/V3, EfficientNet (B0-B7), ConvNeXt, and Xception from `keras.applications`.
- [x] **Zoo Parity (NLP):** Validate BERT, RoBERTa, GPT-2, and Gemma architectures from `Keras-NLP`.
- [x] **Zoo Parity (CV):** Validate YOLOv8, RetinaNet, and Stable Diffusion architectures from `Keras-CV`.
- [x] **Mathematical Tolerance:** Establish CI jobs comparing Keras `model.predict(x)` output vs `onnx9000` (WASM/C++) execution to `< 1e-4` absolute tolerance.
- [x] **Memory Profiling:** Validate WASM SIMD execution memory leak stability on large sequential NLP Keras conversions (e.g., iterative text generation loops).
- [x] **Chrome Tracing:** Generate Chrome `trace.json` files mapping Keras node executions explicitly to WebGPU/WASM microsecond benchmarks.
