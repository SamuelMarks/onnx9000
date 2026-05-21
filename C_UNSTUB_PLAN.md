# Exhaustive C/C++ & WASM Generator Unstub Plan

This document dictates the complete, exhaustive roadmap to fully unstub the ONNX9000 C/C++ and WASM generator pipeline. Every task must adhere to the global project quality mandates.

## 0. Global Quality & Architectural Mandates
These requirements must be validated for every phase and every feature introduced.

- [ ] **100% Test Coverage:** Every new function, branch, and operator implementation must be fully covered by unit tests (AST generation) and E2E validation tests (compiler execution).
- [ ] **100% Doc Coverage:** All TypeScript methods, C/C++ structs, and functions must be exhaustively documented using TSDoc / Doxygen format.
- [ ] **C++ Project Structure:** When generating C++ projects (if `emitCpp` is enabled), place `.hpp` and `.cpp` files in a `src` subdirectory rather than the root directory.
- [ ] **CMake Integration:** Generate a `src/CMakeLists.txt` (following CMake best practices) alongside the root `CMakeLists.txt` for C++ target compilation.
- [ ] **Error Handling (Rust parity):** Ensure any Rust-based backend tooling interacting with this pipeline has one big error enum (with `derive_more`), with strictly zero use of `unwrap` or `anyhow`.

## Phase 1: Core Pipeline Integration (`compileOnnxToC`)
- [x] Remove the hardcoded dummy `[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]` WASM return payload.
- [x] Modify `compileOnnxToC` to deserialize the `Uint8Array` buffer using `@onnx9000/core` `BufferReader`.
- [x] Implement a topological sort over `Graph.nodes` to ensure dependency-ordered execution.
- [x] Implement a static shape inference pass over the topologically sorted graph.
- [x] Instantiate `CGenerator` (or `CppGenerator`) correctly based on `emitCpp` option.
- [x] Calculate the total memory footprint dynamically and accurately populate `generator.generateSummary()`.
- [x] Return the actual concatenated output of `generateHeader()`, `generateSource()`, and `generateCMakeLists()` if applicable.
- [x] Fail gracefully with a structured error if the `Graph` contains unimplemented ops.

## Phase 2: Memory Management & Types
- [x] Define `Tensor` struct representing multi-dimensional data (`rank`, `dims`, `strides`, `data` pointer).
- [x] Implement a pre-calculated static memory arena `float workspace[MAX_MEMORY_FOOTPRINT]`.
- [x] Implement offset mapping logic so each intermediate tensor points to a reused segment in the `workspace` based on liveness analysis.
- [x] Generate dynamic memory allocation pathways (`malloc`/`free`) when statically bounded sizes cannot be determined.
- [x] Ensure all multi-dimensional arrays are properly stride-aligned.
- [x] Implement SIMD memory alignment generation (e.g., `posix_memalign` or `__attribute__((aligned(32)))`).
- [x] Implement broadcasting utilities (scalar-to-tensor, 1D-to-ND, ND-to-ND).

## Phase 3: Exhaustive ONNX Operator Implementations
Each operator requires its own C/C++ generation logic, memory boundary checks, 100% doc coverage, and 100% unit test coverage validating mathematical equivalence.

### Neural Network: Convolutions & Pooling
- [x] `Conv` (1D, 2D, 3D support, handling `pads`, `strides`, `dilations`, `group`).
- [x] `ConvInteger`
- [x] `ConvTranspose`
- [x] `MaxPool`
- [x] `AveragePool`
- [x] `MaxUnpool`
- [x] `GlobalMaxPool`
- [x] `GlobalAveragePool`
- [x] `LpPool`
- [x] `GlobalLpPool`

### Neural Network: Normalization & Padding
- [x] `BatchNormalization` (handling `epsilon`, `momentum`, training mode false).
- [x] `InstanceNormalization`
- [x] `LayerNormalization`
- [x] `GroupNormalization`
- [x] `LocalResponseNormalization` (LRN)
- [x] `Pad`
- [x] `Dropout` (No-op during inference).

### Neural Network: Activations
- [x] `Relu`
- [x] `Sigmoid`
- [x] `Tanh`
- [x] `LeakyReLU`
- [x] `PRelu`
- [x] `Elu`
- [x] `Celu`
- [x] `Selu`
- [x] `Gelu`
- [x] `Softmax`
- [x] `LogSoftmax`
- [x] `Hardmax`
- [x] `HardSigmoid`
- [x] `HardSwish`
- [x] `Softplus`
- [x] `Softsign`
- [x] `Shrink`
- [x] `ThresholdedRelu`

### Math: Element-wise Arithmetic
- [x] `Add`
- [x] `Sub`
- [x] `Mul`
- [x] `Div`
- [x] `Pow`
- [x] `Mod`
- [x] `Abs`
- [x] `Neg`
- [x] `Sign`
- [x] `Exp`
- [x] `Log`
- [x] `Log10`
- [x] `Sqrt`

### Math: Element-wise Trigonometry & Advanced
- [x] `Acos`
- [x] `Acosh`
- [x] `Asin`
- [x] `Asinh`
- [x] `Atan`
- [x] `Atanh`
- [x] `Cos`
- [x] `Cosh`
- [x] `Sin`
- [x] `Sinh`
- [x] `Tan`
- [x] `Erf`
- [x] `Ceil`
- [x] `Floor`
- [x] `Round`
- [x] `Trunc`

### Math: Reductions
- [x] `ReduceMax`
- [x] `ReduceMin`
- [x] `ReduceMean`
- [x] `ReduceSum`
- [x] `ReduceSumSquare`
- [x] `ReduceProd`
- [x] `ReduceL1`
- [x] `ReduceL2`
- [x] `ReduceLogSum`
- [x] `ReduceLogSumExp`

### Linear Algebra
- [x] `Gemm` (handling `alpha`, `beta`, `transA`, `transB`).
- [x] `MatMul` (Batched N-dimensional).
- [x] `MatMulInteger`
- [x] `MatMulInteger16`
- [x] `Det`
- [x] `Trilu`
- [x] `Einsum`

### Logical & Bitwise
- [x] `And`
- [x] `Or`
- [x] `Xor`
- [x] `Not`
- [x] `Equal`
- [x] `Greater`
- [x] `GreaterOrEqual`
- [x] `Less`
- [x] `LessOrEqual`
- [x] `IsNaN`
- [x] `IsInf`
- [x] `BitShift`
- [x] `BitwiseAnd`
- [x] `BitwiseNot`
- [x] `BitwiseOr`
- [x] `BitwiseXor`

### Tensor Manipulation: Shape & Reshaping
- [x] `Reshape` (Implement as zero-copy metadata change where memory layout allows).
- [x] `Flatten`
- [x] `Squeeze`
- [x] `Unsqueeze`
- [x] `Transpose`
- [x] `Shape`
- [x] `Size`
- [x] `Cast`
- [x] `CastLike`

### Tensor Manipulation: Slicing, Gathering & Scattering
- [x] `Concat`
- [x] `Split`
- [x] `Slice`
- [x] `Gather`
- [x] `GatherElements`
- [x] `GatherND`
- [x] `Scatter`
- [x] `ScatterElements`
- [x] `ScatterND`
- [x] `Tile`
- [x] `Expand`
- [x] `Compress`
- [x] `NonZero`
- [x] `Identity`

### Constant Generation
- [x] `Constant`
- [x] `ConstantOfShape`
- [x] `EyeLike`

### Control Flow & Sequence
- [x] `If`
- [x] `Loop`
- [x] `Scan`
- [x] `SequenceAt`
- [x] `SequenceConstruct`
- [x] `SequenceEmpty`
- [x] `SequenceErase`
- [x] `SequenceInsert`
- [x] `SequenceLength`
- [x] `ReverseSequence`

### RNN / Time Series
- [x] `RNN`
- [x] `LSTM`
- [x] `GRU`

### Vision & Object Detection
- [x] `NonMaxSuppression`
- [x] `RoiAlign`
- [x] `MaxRoiPool`
- [x] `GridSample`
- [x] `Resize`
- [x] `CenterCrop`
- [x] `SpaceToDepth`
- [x] `DepthToSpace`

### Quantization
- [x] `QuantizeLinear`
- [x] `DequantizeLinear`
- [x] `DynamicQuantizeLinear`
- [x] `QLinearConv`
- [x] `QLinearMatMul`

## Phase 4: Weights Serialization & Formatting
- [x] Extract graph `initializer` arrays into a standardized byte format.
- [x] Generate a `model_weights.bin` output artifact to avoid massive C-file sizes.
- [x] Ensure Endian-agnostic generation and reading of `.bin` weights.
- [x] Generate C function `load_weights(const char* path)` utilizing `fread` for runtime population.
- [x] Implement FP16/INT8 dequantization hooks at the loader level to minimize disk footprint.
- [x] Guarantee 100% test coverage of the serialization and loading mechanism.

## Phase 5: WASM Native Integration (`wasm-compiler`)
- [x] Integrate a real WebAssembly emitter (e.g., `binaryen` or `wabt`) instead of the hardcoded 8-byte array.
- [x] Translate ONNX IR directly into WASM opcode generation (or pipe the generated C through an Emscripten target in memory).
- [x] Allocate WASM `Memory` pages dynamically based on the inferred static memory bounds.
- [x] Expose JS-to-WASM bindings to write input tensors to WASM memory boundaries safely.
- [x] Ensure 100% documentation for the public `WasmCompiler` SDK methods.

## Phase 6: End-to-End Validation
- [x] Expand Playwright E2E tests (`e2e/wasm-demo.spec.ts`) to intercept Keras/ONNX generation and compile the actual C output using `gcc`.
- [x] Execute `a.out` inside the E2E test on realistic input tensors and assert against `onnxruntime` outputs.
- [x] Measure execution performance and memory footprint as part of CI reporting.
- [x] Use `AddressSanitizer` in the C execution step of the E2E tests to validate 100% memory safety.