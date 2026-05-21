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
- [ ] Remove the hardcoded dummy `[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]` WASM return payload.
- [ ] Modify `compileOnnxToC` to deserialize the `Uint8Array` buffer using `@onnx9000/core` `BufferReader`.
- [ ] Implement a topological sort over `Graph.nodes` to ensure dependency-ordered execution.
- [ ] Implement a static shape inference pass over the topologically sorted graph.
- [ ] Instantiate `CGenerator` (or `CppGenerator`) correctly based on `emitCpp` option.
- [ ] Calculate the total memory footprint dynamically and accurately populate `generator.generateSummary()`.
- [ ] Return the actual concatenated output of `generateHeader()`, `generateSource()`, and `generateCMakeLists()` if applicable.
- [ ] Fail gracefully with a structured error if the `Graph` contains unimplemented ops.

## Phase 2: Memory Management & Types
- [ ] Define `Tensor` struct representing multi-dimensional data (`rank`, `dims`, `strides`, `data` pointer).
- [ ] Implement a pre-calculated static memory arena `float workspace[MAX_MEMORY_FOOTPRINT]`.
- [ ] Implement offset mapping logic so each intermediate tensor points to a reused segment in the `workspace` based on liveness analysis.
- [ ] Generate dynamic memory allocation pathways (`malloc`/`free`) when statically bounded sizes cannot be determined.
- [ ] Ensure all multi-dimensional arrays are properly stride-aligned.
- [ ] Implement SIMD memory alignment generation (e.g., `posix_memalign` or `__attribute__((aligned(32)))`).
- [ ] Implement broadcasting utilities (scalar-to-tensor, 1D-to-ND, ND-to-ND).

## Phase 3: Exhaustive ONNX Operator Implementations
Each operator requires its own C/C++ generation logic, memory boundary checks, 100% doc coverage, and 100% unit test coverage validating mathematical equivalence.

### Neural Network: Convolutions & Pooling
- [ ] `Conv` (1D, 2D, 3D support, handling `pads`, `strides`, `dilations`, `group`).
- [ ] `ConvInteger`
- [ ] `ConvTranspose`
- [ ] `MaxPool`
- [ ] `AveragePool`
- [ ] `MaxUnpool`
- [ ] `GlobalMaxPool`
- [ ] `GlobalAveragePool`
- [ ] `LpPool`
- [ ] `GlobalLpPool`

### Neural Network: Normalization & Padding
- [ ] `BatchNormalization` (handling `epsilon`, `momentum`, training mode false).
- [ ] `InstanceNormalization`
- [ ] `LayerNormalization`
- [ ] `GroupNormalization`
- [ ] `LocalResponseNormalization` (LRN)
- [ ] `Pad`
- [ ] `Dropout` (No-op during inference).

### Neural Network: Activations
- [ ] `Relu`
- [ ] `Sigmoid`
- [ ] `Tanh`
- [ ] `LeakyReLU`
- [ ] `PRelu`
- [ ] `Elu`
- [ ] `Celu`
- [ ] `Selu`
- [ ] `Gelu`
- [ ] `Softmax`
- [ ] `LogSoftmax`
- [ ] `Hardmax`
- [ ] `HardSigmoid`
- [ ] `HardSwish`
- [ ] `Softplus`
- [ ] `Softsign`
- [ ] `Shrink`
- [ ] `ThresholdedRelu`

### Math: Element-wise Arithmetic
- [ ] `Add`
- [ ] `Sub`
- [ ] `Mul`
- [ ] `Div`
- [ ] `Pow`
- [ ] `Mod`
- [ ] `Abs`
- [ ] `Neg`
- [ ] `Sign`
- [ ] `Exp`
- [ ] `Log`
- [ ] `Log10`
- [ ] `Sqrt`

### Math: Element-wise Trigonometry & Advanced
- [ ] `Acos`
- [ ] `Acosh`
- [ ] `Asin`
- [ ] `Asinh`
- [ ] `Atan`
- [ ] `Atanh`
- [ ] `Cos`
- [ ] `Cosh`
- [ ] `Sin`
- [ ] `Sinh`
- [ ] `Tan`
- [ ] `Erf`
- [ ] `Ceil`
- [ ] `Floor`
- [ ] `Round`
- [ ] `Trunc`

### Math: Reductions
- [ ] `ReduceMax`
- [ ] `ReduceMin`
- [ ] `ReduceMean`
- [ ] `ReduceSum`
- [ ] `ReduceSumSquare`
- [ ] `ReduceProd`
- [ ] `ReduceL1`
- [ ] `ReduceL2`
- [ ] `ReduceLogSum`
- [ ] `ReduceLogSumExp`

### Linear Algebra
- [ ] `Gemm` (handling `alpha`, `beta`, `transA`, `transB`).
- [ ] `MatMul` (Batched N-dimensional).
- [ ] `MatMulInteger`
- [ ] `MatMulInteger16`
- [ ] `Det`
- [ ] `Trilu`
- [ ] `Einsum`

### Logical & Bitwise
- [ ] `And`
- [ ] `Or`
- [ ] `Xor`
- [ ] `Not`
- [ ] `Equal`
- [ ] `Greater`
- [ ] `GreaterOrEqual`
- [ ] `Less`
- [ ] `LessOrEqual`
- [ ] `IsNaN`
- [ ] `IsInf`
- [ ] `BitShift`
- [ ] `BitwiseAnd`
- [ ] `BitwiseNot`
- [ ] `BitwiseOr`
- [ ] `BitwiseXor`

### Tensor Manipulation: Shape & Reshaping
- [ ] `Reshape` (Implement as zero-copy metadata change where memory layout allows).
- [ ] `Flatten`
- [ ] `Squeeze`
- [ ] `Unsqueeze`
- [ ] `Transpose`
- [ ] `Shape`
- [ ] `Size`
- [ ] `Cast`
- [ ] `CastLike`

### Tensor Manipulation: Slicing, Gathering & Scattering
- [ ] `Concat`
- [ ] `Split`
- [ ] `Slice`
- [ ] `Gather`
- [ ] `GatherElements`
- [ ] `GatherND`
- [ ] `Scatter`
- [ ] `ScatterElements`
- [ ] `ScatterND`
- [ ] `Tile`
- [ ] `Expand`
- [ ] `Compress`
- [ ] `NonZero`
- [ ] `Identity`

### Constant Generation
- [ ] `Constant`
- [ ] `ConstantOfShape`
- [ ] `EyeLike`

### Control Flow & Sequence
- [ ] `If`
- [ ] `Loop`
- [ ] `Scan`
- [ ] `SequenceAt`
- [ ] `SequenceConstruct`
- [ ] `SequenceEmpty`
- [ ] `SequenceErase`
- [ ] `SequenceInsert`
- [ ] `SequenceLength`
- [ ] `ReverseSequence`

### RNN / Time Series
- [ ] `RNN`
- [ ] `LSTM`
- [ ] `GRU`

### Vision & Object Detection
- [ ] `NonMaxSuppression`
- [ ] `RoiAlign`
- [ ] `MaxRoiPool`
- [ ] `GridSample`
- [ ] `Resize`
- [ ] `CenterCrop`
- [ ] `SpaceToDepth`
- [ ] `DepthToSpace`

### Quantization
- [ ] `QuantizeLinear`
- [ ] `DequantizeLinear`
- [ ] `DynamicQuantizeLinear`
- [ ] `QLinearConv`
- [ ] `QLinearMatMul`

## Phase 4: Weights Serialization & Formatting
- [ ] Extract graph `initializer` arrays into a standardized byte format.
- [ ] Generate a `model_weights.bin` output artifact to avoid massive C-file sizes.
- [ ] Ensure Endian-agnostic generation and reading of `.bin` weights.
- [ ] Generate C function `load_weights(const char* path)` utilizing `fread` for runtime population.
- [ ] Implement FP16/INT8 dequantization hooks at the loader level to minimize disk footprint.
- [ ] Guarantee 100% test coverage of the serialization and loading mechanism.

## Phase 5: WASM Native Integration (`wasm-compiler`)
- [ ] Integrate a real WebAssembly emitter (e.g., `binaryen` or `wabt`) instead of the hardcoded 8-byte array.
- [ ] Translate ONNX IR directly into WASM opcode generation (or pipe the generated C through an Emscripten target in memory).
- [ ] Allocate WASM `Memory` pages dynamically based on the inferred static memory bounds.
- [ ] Expose JS-to-WASM bindings to write input tensors to WASM memory boundaries safely.
- [ ] Ensure 100% documentation for the public `WasmCompiler` SDK methods.

## Phase 6: End-to-End Validation
- [ ] Expand Playwright E2E tests (`e2e/wasm-demo.spec.ts`) to intercept Keras/ONNX generation and compile the actual C output using `gcc`.
- [ ] Execute `a.out` inside the E2E test on realistic input tensors and assert against `onnxruntime` outputs.
- [ ] Measure execution performance and memory footprint as part of CI reporting.
- [ ] Use `AddressSanitizer` in the C execution step of the E2E tests to validate 100% memory safety.