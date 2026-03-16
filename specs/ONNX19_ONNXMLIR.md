# onnx-mlir Replication & Parity Tracker

## Description
This document tracks the complete reimplementation of `onnx-mlir` (Ahead-Of-Time Compilation) within the `onnx9000` ecosystem.
The standard `onnx-mlir` project relies on the massive LLVM compiler infrastructure and the MLIR (Multi-Level Intermediate Representation) dialect framework. Building and utilizing it requires a massive C++ toolchain.
Our `onnx9000` reimplementation completely bypasses LLVM/MLIR. Instead, it transpiles the pure-Python `core.ir` directly into highly optimized, strict C++23 source files using Jinja2 templates. This C++ can then be compiled natively via `g++`/`clang++` for zero-overhead server execution, or via `emcc` (Emscripten) into microscopic standalone WebAssembly (`.wasm`) payloads that require *zero external ML runtimes* (like ONNX Runtime Web) to execute in the browser.

## Exhaustive Parity Checklist

### 1. Codegen Architecture & Static Memory Arena (40+ items)
- [ ] Implement C++23 code generation engine using Jinja2 templates
- [ ] Implement static shape resolution pass ahead of transpilation
- [ ] Implement static dtype resolution pass ahead of transpilation
- [ ] Implement global contiguous Memory Arena calculator for all intermediate tensors
- [ ] Eliminate all dynamic `malloc`/`new` and `free`/`delete` calls during inference execution
- [ ] Pre-calculate and hardcode exact byte offsets for every intermediate tensor in the Arena
- [ ] Support generating isolated C++ execution functions (e.g., `execute_model()`)
- [ ] Support generating C++ classes encapsulating the model state
- [ ] Embed static `Constant` tensors directly into the C++ binary (arrays)
- [ ] Support writing `Constant` tensors to an external `.bin` file loaded at runtime via `mmap`
- [ ] Implement `std::expected` (C++23) for monadic error handling boundaries
- [ ] Ensure all generated kernel executions are marked `noexcept`
- [ ] Support loop unrolling natively via C++ `#pragma unroll` injections
- [ ] Emit strict `#pragma omp parallel for` (OpenMP) directives for multi-threading target
- [ ] Emit specific `#pragma clang loop vectorize(enable)` (LLVM/Clang) directives
- [ ] Support compiling to standalone shared libraries (`.so` / `.dylib` / `.dll`)
- [ ] Support compiling to standalone static libraries (`.a` / `.lib`)
- [ ] Implement `pybind11` bridge generation for instantly loading the `.so` back into Python
- [ ] Expose native C ABI (`extern "C"`) functions for generic FFI bindings (Rust, Go, etc.)
- [ ] Emit zero-copy array pointers across the C API boundary
- [ ] Validate generated C++ logic statically via `static_assert` statements
- [ ] Emulate multidimensional arrays using contiguous flat `std::span` or `std::vector` abstractions
- [ ] Implement broadcasting math macros statically in C++
- [ ] Implement N-dimensional indexing macros statically (`index_4d(n, c, h, w)`)
- [ ] Ensure strict adherence to `-Wall -Wextra -Werror` warnings
- [ ] Optimize arithmetic precision (avoiding implicit `double` promotion in C++)
- [ ] Support `__restrict__` pointers for alias-free compiler optimizations
- [ ] Handle subgraphs (`If`, `Loop`) via recursive C++ function generation
- [ ] Support generating static switch statements for categorical routing
- [ ] Generate metadata accessors (`get_input_shape()`, `get_output_type()`)
- [ ] Eliminate dead C++ variables (transpiler level DCE)
- [ ] Optimize identical loop fusion directly in the C++ generator

### 2. WebAssembly (WASM) Backend Compilation (30+ items)
- [ ] Detect Emscripten (`emcc`) installation automatically
- [ ] Emit Emscripten JS glue code (`--bind` or WebIDL) automatically
- [ ] Compile directly to `.wasm` payload
- [ ] Compile to combined `.js` and `.wasm` standard module format
- [ ] Support building strictly standalone WASM (no JS glue, using pure Wasm imports/exports)
- [ ] Enable `-O3` Emscripten optimization flags by default
- [ ] Enable `-Os` (size optimization) flag
- [ ] Enable `-Oz` (extreme size optimization) flag
- [ ] Inject `-msimd128` flags natively for WebAssembly SIMD
- [ ] Emit explicit WASM SIMD intrinsics (`wasm_simd128.h`) for heavy math kernels
- [ ] Ensure the WASM payload fits within standard browser limits (without breaking WebAssembly.instantiate limits)
- [ ] Provide configurable WASM `INITIAL_MEMORY` parameters based on the calculated static Arena size
- [ ] Provide configurable WASM `MAXIMUM_MEMORY` parameters
- [ ] Enable `ALLOW_MEMORY_GROWTH=0` when shapes are perfectly static (performance boost)
- [ ] Expose JS typed-array bridges for zero-copy input/output evaluation
- [ ] Transpile JS `Float32Array` directly to the `execute` C++ pointers
- [ ] Compile WebWorker wrappers automatically for off-main-thread browser execution
- [ ] Support multithreading in WASM via `USE_PTHREADS=1` and `SharedArrayBuffer`
- [ ] Compress large constants externally for HTTP chunking alongside the `.wasm`
- [ ] Provide TypeScript definitions (`.d.ts`) for the generated WASM module natively
- [ ] Verify execution exactly matches Python ONNX predictions (browser unit tests)
- [ ] Support `Node.js` environment natively in the generated WASM wrappers
- [ ] Support `Deno` environment natively in the generated WASM wrappers

### 3. CPU Core Operations (C++ Kernels) (40+ items)
- [ ] Implement `Add` kernel (broadcasted and flat)
- [ ] Implement `Sub` kernel
- [ ] Implement `Mul` kernel
- [ ] Implement `Div` kernel
- [ ] Implement `MatMul` kernel (Naive 3-loop)
- [ ] Implement `MatMul` kernel (Cache-blocked / Tiled)
- [ ] Implement `Conv` kernel (Naive im2col + gemm)
- [ ] Implement `Conv` kernel (Direct spatial convolution)
- [ ] Implement `Conv` kernel (Depthwise specific optimization)
- [ ] Implement `MaxPool` kernel
- [ ] Implement `AveragePool` kernel
- [ ] Implement `GlobalAveragePool` kernel
- [ ] Implement `Relu` kernel (branchless `std::max`)
- [ ] Implement `LeakyRelu` kernel
- [ ] Implement `Sigmoid` kernel (using fast math approximations if enabled)
- [ ] Implement `Tanh` kernel
- [ ] Implement `Exp` kernel
- [ ] Implement `Log` kernel
- [ ] Implement `Softmax` kernel (numerically stable: subtract max)
- [ ] Implement `ReduceSum` kernel
- [ ] Implement `ReduceMean` kernel
- [ ] Implement `ReduceMax` kernel
- [ ] Implement `ReduceMin` kernel
- [ ] Implement `Transpose` kernel
- [ ] Implement `Reshape` (No-op in flat memory, pure logical remap)
- [ ] Implement `Flatten` (No-op)
- [ ] Implement `Squeeze` (No-op)
- [ ] Implement `Unsqueeze` (No-op)
- [ ] Implement `Concat` kernel
- [ ] Implement `Split` kernel
- [ ] Implement `Slice` kernel
- [ ] Implement `Gather` kernel
- [ ] Implement `ScatterElements` kernel
- [ ] Implement `ScatterND` kernel
- [ ] Implement `GatherND` kernel
- [ ] Implement `Where` kernel
- [ ] Implement `Cast` kernel
- [ ] Implement `ConstantOfShape` kernel (memset)
- [ ] Implement `Pad` kernel (constant padding)
- [ ] Implement `NonMaxSuppression` kernel

### 4. Apple Accelerate Framework Integration (20+ items)
- [ ] Detect `Accelerate` framework on macOS natively
- [ ] Bind `MatMul` to `cblas_sgemm` (Float32)
- [ ] Bind `MatMul` to `cblas_dgemm` (Float64)
- [ ] Bind `MatMul` to `cblas_hgemm` (Float16 if supported)
- [ ] Bind Elementwise `Add` to `vDSP_vadd`
- [ ] Bind Elementwise `Mul` to `vDSP_vmul`
- [ ] Bind Elementwise `Div` to `vDSP_vdiv`
- [ ] Bind `Exp` to `vforce_vexp` / `vvexpf`
- [ ] Bind `Log` to `vforce_vlog` / `vvlogf`
- [ ] Bind `Sin` to `vforce_vsin` / `vvsinf`
- [ ] Bind `Cos` to `vforce_vcos` / `vvcosf`
- [ ] Bind `Tanh` to `vforce_vtanh` / `vvtanhf`
- [ ] Bind `Sqrt` to `vforce_vsqrt` / `vvsqrtf`
- [ ] Bind `ReduceSum` to `vDSP_sve`
- [ ] Bind `ReduceMax` to `vDSP_maxv`
- [ ] Bind `ReduceMin` to `vDSP_minv`
- [ ] Bind `ReduceMean` to `vDSP_meanv`
- [ ] Validate zero-copy passing of memory arena pointers to `cblas`
- [ ] Dynamically link `-framework Accelerate` during `clang++` compilation
- [ ] Fallback to native C++ loop if dimensions do not match BLAS requirements

### 5. OpenBLAS & MKL Fallback (15+ items)
- [ ] Detect OpenBLAS on Linux/Windows natively
- [ ] Bind `MatMul` to OpenBLAS `cblas_sgemm`
- [ ] Detect Intel MKL on compatible hardware
- [ ] Bind `MatMul` to Intel MKL `cblas_sgemm`
- [ ] Support dynamic linking of `libopenblas.so` during compilation
- [ ] Support static linking of OpenBLAS
- [ ] Validate row-major vs col-major transpose flags (`CblasRowMajor`) for MKL
- [ ] Inject `#include <cblas.h>` or `<mkl.h>` dynamically based on target flag
- [ ] Fallback gracefully to cache-blocked C++ kernels if no BLAS is detected

### 6. Neural Architecture & Optimization Specifics (25+ items)
- [ ] Implement `LayerNormalization` kernel natively in C++
- [ ] Implement `BatchNormalization` kernel natively (Inference mode)
- [ ] Implement `Gelu` kernel natively (Erf and Tanh approximations)
- [ ] Implement `HardSwish` kernel
- [ ] Implement `Mish` kernel
- [ ] Compile `TreeEnsembleClassifier` natively into static C++ `if/else` bounds or loop structures
- [ ] Compile `TreeEnsembleRegressor` natively
- [ ] Support dynamic sequence execution (`RNN`, `LSTM`, `GRU`) via static unrolling if `seq_len` is constant
- [ ] Support dynamic sequence execution via C++ `for` loops if `seq_len` is dynamic
- [ ] Embed explicit lookup tables (LUTs) for complex math if requested (`--fast-math`)
- [ ] Provide exact memory strides mathematically for N-Dimensional slicing without looping
- [ ] Translate `ai.onnx.ml.Scaler` to a vectorized loop
- [ ] Translate `ai.onnx.ml.OneHotEncoder` to explicit array indexing
- [ ] Handle `Einsum` statically if equation is solvable at compile time
- [ ] Generate standard C++ `<random>` library calls for `RandomUniform`
- [ ] Generate standard C++ `<random>` library calls for `RandomNormal`
- [ ] Generate standard C++ calls for `Multinomial`
- [ ] Handle explicit memory `memset` for `ZerosLike`
- [ ] Ensure `Softmax` operations are cache-friendly (processing rows continuously)
- [ ] Strip out `Dropout` operations entirely during C++ generation (inference mode)
- [ ] Strip out `Identity` operations natively during the transpiler phase
- [ ] Validate `Cast` kernels correctly map `float` to `int` safely

### 7. Explicit Advanced C++ Transpiler Support (40+ items)
- [ ] Support `float16` (`_Float16`) code generation natively in C++23
- [ ] Support `bfloat16` (`__bf16`) code generation natively
- [ ] Support `int8_t` memory alignment natively
- [ ] Support `uint8_t` memory alignment natively
- [ ] Support `int64_t` processing safely (preventing `int32` overflows in loops)
- [ ] Generate pure `<complex>` headers for ONNX complex math operations
- [ ] Implement `std::string` handling for ONNX `String` tensors in the C++ backend
- [ ] Extract literal dimensions to explicit `constexpr` variables
- [ ] Map Python strings to `constexpr std::string_view` mapping tables
- [ ] Expose native C++ `execute(const float* input, float* output)` function signatures
- [ ] Manage dynamically shaped inputs via pointer sizes `execute(float* in, size_t dim)`
- [ ] Use `std::unique_ptr` for memory arena to prevent leaks if allocated dynamically
- [ ] Handle `ConstantOfShape` with dynamic shapes via `std::vector` inside the Arena wrapper
- [ ] Generate specific `Makefile` or `CMakeLists.txt` (optional) alongside the `.cpp` file
- [ ] Execute `clang-format` automatically on generated C++ to maintain extreme readability
- [ ] Auto-generate a `main.cpp` entrypoint for direct CLI testing/benchmarking of the compiled model
- [ ] Generate internal C++ benchmarking macros (`#define PROFILE_LAYERS`) to time individual kernels
- [ ] Ensure strict adherence to `clang-tidy` constraints
- [ ] Replace `std::pow(x, 2)` with `x * x` explicitly for performance
- [ ] Optimize division by powers of 2 into right-shifts (`>>`) for integers
- [ ] Generate explicit branch-prediction hints (`[[likely]]`, `[[unlikely]]`) for `If` statements
- [ ] Embed Model Name, Version, and Producer as `#define` strings
- [ ] Extract and embed the original ONNX `doc_string` as a C++ multiline comment

### 8. Testing & Validation (Edge Cases) (30+ items)
- [ ] Unit Test: Compile pure `Add` graph to C++ and execute via Pybind11
- [ ] Unit Test: Compile `MatMul` (statically shaped) and execute natively
- [ ] Unit Test: Compile `MatMul` (dynamic batch size) and execute natively
- [ ] Unit Test: Compile `Conv` + `Relu` chain and validate output against ONNX Runtime (atol=1e-5)
- [ ] Unit Test: Compile standard ResNet50 to C++ and evaluate ImageNet sample
- [ ] Unit Test: Compile massive `TreeEnsemble` (Random Forest) to standalone WASM (<1MB payload)
- [ ] Unit Test: Execute WASM binary strictly inside V8 (Node.js) and validate results
- [ ] Unit Test: Execute `If` branching structures generated as C++ `if/else`
- [ ] Unit Test: Validate loop unrolling limits gracefully (falling back to standard loops for large N)
- [ ] Validate static Arena calculator correctly aliases memory perfectly across sequential layers
- [ ] Catch dynamic shape violations at transpilation time
- [ ] Unit Test: Transpile and link Apple Accelerate strictly on MacOS and benchmark
- [ ] Validate OpenBLAS linkage on Ubuntu environments
- [ ] Stress Test: Compile a 1000-layer generated graph (testing Jinja2 stack depth limits)
- [ ] Ensure extreme model topologies do not cause C++ compiler OOM (Out Of Memory)
- [ ] Test cross-compilation (e.g. compiling for `aarch64-linux-gnu` from x86_64) if LLVM/Clang is used natively
- [ ] Validate WASM SIMD execution strictly in Chrome


### 9. Exhaustive C++ Operator Implementations (60+ items)
- [ ] Implement `Abs` kernel (branchless `std::abs`)
- [ ] Implement `Acos` kernel (`std::acos`)
- [ ] Implement `Acosh` kernel (`std::acosh`)
- [ ] Implement `Add` kernel (with explicit 1D, 2D, 3D, 4D broadcasting loops)
- [ ] Implement `And` kernel
- [ ] Implement `ArgMax` kernel
- [ ] Implement `ArgMin` kernel
- [ ] Implement `Asin` kernel (`std::asin`)
- [ ] Implement `Asinh` kernel (`std::asinh`)
- [ ] Implement `Atan` kernel (`std::atan`)
- [ ] Implement `Atanh` kernel (`std::atanh`)
- [ ] Implement `BitShift` kernel (`<<`, `>>`)
- [ ] Implement `BitwiseAnd` kernel (`&`)
- [ ] Implement `BitwiseNot` kernel (`~`)
- [ ] Implement `BitwiseOr` kernel (`|`)
- [ ] Implement `BitwiseXor` kernel (`^`)
- [ ] Implement `Ceil` kernel (`std::ceil`)
- [ ] Implement `Clip` kernel (`std::clamp`)
- [ ] Implement `Compress` kernel
- [ ] Implement `Constant` kernel (memcpy from ROM)
- [ ] Implement `Cos` kernel (`std::cos`)
- [ ] Implement `Cosh` kernel (`std::cosh`)
- [ ] Implement `CumSum` kernel
- [ ] Implement `DepthToSpace` kernel (memory permutation)
- [ ] Implement `DequantizeLinear` kernel
- [ ] Implement `Det` kernel
- [ ] Implement `Dropout` kernel (inference mode - pass-through)
- [ ] Implement `Einsum` kernel (naive nested loops based on generated string)
- [ ] Implement `Elu` kernel
- [ ] Implement `Equal` kernel (`==`)
- [ ] Implement `Erf` kernel (`std::erf`)
- [ ] Implement `Expand` kernel (logical broadcast remap)
- [ ] Implement `EyeLike` kernel
- [ ] Implement `Floor` kernel (`std::floor`)
- [ ] Implement `GatherElements` kernel
- [ ] Implement `GlobalLpPool` kernel
- [ ] Implement `GlobalMaxPool` kernel
- [ ] Implement `Greater` kernel (`>`)
- [ ] Implement `GreaterOrEqual` kernel (`>=`)
- [ ] Implement `Hardmax` kernel
- [ ] Implement `HardSigmoid` kernel
- [ ] Implement `Identity` kernel (if not stripped by DCE)
- [ ] Implement `IsInf` kernel (`std::isinf`)
- [ ] Implement `IsNaN` kernel (`std::isnan`)
- [ ] Implement `LRN` kernel (Local Response Normalization)
- [ ] Implement `Less` kernel (`<`)
- [ ] Implement `LessOrEqual` kernel (`<=`)
- [ ] Implement `LogSoftmax` kernel
- [ ] Implement `LpNormalization` kernel
- [ ] Implement `LpPool` kernel
- [ ] Implement `Max` kernel (`std::max`)
- [ ] Implement `MaxRoiPool` kernel
- [ ] Implement `Mean` kernel (Elementwise mean)
- [ ] Implement `Min` kernel (`std::min`)
- [ ] Implement `Mod` kernel (`std::fmod` / `%`)
- [ ] Implement `Multinomial` kernel (using `<random>`)
- [ ] Implement `Neg` kernel (`-`)
- [ ] Implement `NonZero` kernel (Dynamic memory allocation required)
- [ ] Implement `Not` kernel (`!`)
- [ ] Implement `OneHot` kernel
- [ ] Implement `Or` kernel (`||`)

### 10. Memory Planning & Dynamic Allocations (30+ items)
- [ ] Implement dynamic tensor memory reallocation gracefully (for `NonZero` and `Compress`)
- [ ] Support falling back from Static Arena to `std::vector` if fully dynamic shapes are encountered
- [ ] Provide explicit C++ `Context` struct tracking dynamic sizes at runtime
- [ ] Provide explicit C++ `Allocator` interface for custom memory management integration
- [ ] Optimize intermediate buffer reuse (e.g. `Buffer A` -> `Buffer B` -> `Buffer A`) via graph coloring
- [ ] Validate memory graph coloring via static verification scripts
- [ ] Support generating `alignas(64)` for strict cache-line alignment natively
- [ ] Support generating `alignas(32)` for AVX instructions specifically
- [ ] Guarantee `alignas(16)` for WebAssembly SIMD boundaries
- [ ] Support `#pragma pack` for compacting structural data definitions
- [ ] Export structural map of the arena layout natively to a JSON descriptor
- [ ] Pre-calculate all broadcasting strides statically into `constexpr` tables
- [ ] Eliminate dynamic stride multiplication in hot loops using pointer arithmetic directly
- [ ] Utilize C++ `std::span` bounds checking if compiled with `-DDEBUG`
- [ ] Disable all bounds checking implicitly under `-O3 -DNDEBUG`
- [ ] Support generating explicit boundary checks for `Gather` (preventing segfaults)
- [ ] Embed external weights (`.bin`) via cross-platform POSIX `mmap()` automatically
- [ ] Embed external weights natively using Windows `CreateFileMapping` / `MapViewOfFile`
- [ ] Optimize integer division using `libdivide` algorithms (statically generated) if divisor is constant
- [ ] Calculate specific memory layout bytes mathematically given tensor shapes and dtypes natively

### 11. Pybind11 & C API Interop (20+ items)
- [ ] Generate strict `<pybind11/pybind11.h>` headers and modules dynamically
- [ ] Generate `<pybind11/numpy.h>` bridges automatically for `py::array_t<float>` handling
- [ ] Guarantee zero-copy evaluation when Python arrays are strictly `C_CONTIGUOUS`
- [ ] Safely copy arrays using C++ `memcpy` if Python inputs are fragmented or `F_CONTIGUOUS`
- [ ] Extract pointer addresses dynamically from `py::buffer_info`
- [ ] Release Python GIL (`py::gil_scoped_release`) natively before invoking the C++ Arena execution
- [ ] Reacquire GIL safely before returning outputs to Python
- [ ] Support compiling the generated `_model.cpp` dynamically inside Python using `subprocess`
- [ ] Load the compiled `.so` using Python `ctypes` or `importlib` dynamically after JIT compilation
- [ ] Generate standard C `extern` functions for C# / .NET P/Invoke integrations
- [ ] Generate standard C `extern` functions for Go / CGO integrations
- [ ] Generate standard C `extern` functions for Rust FFI bindings
- [ ] Export model signature (`get_input_count`, `get_input_name`, `get_output_shape`) via C API
- [ ] Catch native C++ exceptions `try/catch` and translate to Python `RuntimeError` securely

### 12. Advanced Emscripten & WASM Opts (20+ items)
- [ ] Emit `--no-entry` flag dynamically if compiling a pure library (no main)
- [ ] Emit `-s EXPORTED_FUNCTIONS=['_execute', '_malloc', '_free']` automatically
- [ ] Emit `-s EXPORTED_RUNTIME_METHODS=['ccall', 'cwrap']` for dynamic invocation
- [ ] Embed specific WASM `Module` initialization handlers dynamically into the emitted JS
- [ ] Expose `HEAPF32` / `HEAPU8` array views seamlessly back to Javascript natively
- [ ] Handle 64-bit integer variables in JS (using `BigInt` safely over the WASM boundary)
- [ ] Test the execution latency difference of `-O3` vs `-Oz` specifically for CNN models
- [ ] Test the file size difference of `-O3` vs `-Oz` specifically for deep tree ensembles
- [ ] Evaluate `-s ALLOW_MEMORY_GROWTH=1` overhead impact
- [ ] Optimize specific `Math.exp()` / `Math.log()` calls using fast-math if `-O3` is specified
- [ ] Handle WebAssembly Out-Of-Bounds memory traps safely by surfacing Javascript Errors
- [ ] Benchmark WASM `MatMul` speed scaling relative to input dimensions
- [ ] Profile WASM execution without SIMD (Baseline tests)
- [ ] Profile WASM execution with SIMD enabled (Performance tests)


### 13. Opset Compliance & Edge Cases (25+ items)
- [ ] Implement `Pow` kernel (`std::pow`)
- [ ] Implement `PRelu` kernel
- [ ] Implement `QLinearConv` kernel (handling zero-points and scales)
- [ ] Implement `QLinearMatMul` kernel (handling zero-points and scales)
- [ ] Implement `QuantizeLinear` kernel
- [ ] Implement `RNN` kernel (with explicit unrolling options)
- [ ] Implement `RandomNormal` kernel
- [ ] Implement `RandomNormalLike` kernel
- [ ] Implement `RandomUniform` kernel
- [ ] Implement `RandomUniformLike` kernel
- [ ] Implement `Range` kernel
- [ ] Implement `Reciprocal` kernel (`1.0 / x`)
- [ ] Implement `ReduceL1` kernel
- [ ] Implement `ReduceL2` kernel
- [ ] Implement `ReduceLogSum` kernel
- [ ] Implement `ReduceLogSumExp` kernel
- [ ] Implement `ReduceProd` kernel
- [ ] Implement `ReduceSumSquare` kernel
- [ ] Handle unrolling depth limits natively in the C++ generator (preventing massive binary bloat)
- [ ] Warn if `Loop` iterations are highly dynamic (generating `while` loops instead of `for`)
- [ ] Verify execution exactly matches Python ONNX predictions (browser unit tests)
- [ ] Compile and run `onnxruntime` standard compliance models (`test_add`, `test_matmul`) natively
- [ ] Compile and run `onnxruntime` compliance models for CNNs (`test_resnet`) natively
- [ ] Compile and run `onnxruntime` compliance models for NLP (`test_bert`) natively
- [ ] Automatically fallback to executing Python IR if C++ compilation fails locally
