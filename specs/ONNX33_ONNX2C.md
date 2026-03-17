# ONNX33: onnx2c (Web-Native TinyML & Embedded C99 Generator)

## Original Project Description

`onnx2c` (and similar projects like `deepC`) are compilers that parse ONNX models and emit pure, standalone C/C++ source code. This is a game-changer for the TinyML and embedded systems ecosystems (Arduino, ESP32, STM32, Raspberry Pi Pico) because it completely eliminates the need for an operating system, a dynamic memory allocator (malloc/free), or a bulky deep learning runtime library. However, standard `onnx2c` tools are typically written in C++ using heavy LLVM or ONNX Protobuf libraries, requiring a complex local build environment just to translate the model.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

`onnx9000.onnx2c` brings embedded C code generation directly into the browser and Node.js using **pure TypeScript and Python AST transpilation**.

- **Browser-Based C99 Transpilation:** A user can drag an ONNX file into a web page, and the application instantly parses the model, performs static memory planning, and generates a `.c` and `.h` file pair in milliseconds, completely client-side.
- **Zero Dynamic Memory (No `malloc`):** Uses `onnx9000`'s advanced static memory arena planner to hardcode every tensor byte-offset directly into the C99 code, guaranteeing execution safety on microcontrollers with 16KB-256KB of RAM.
- **Hardware Intrinsic Injection:** Rather than generating only naive C loops, the web compiler can optionally inject `#ifdef` blocks targeting ARM CMSIS-NN or Espressif ESP-NN DSP intrinsics for massive acceleration on microcontrollers.
- **Flash Storage Optimization:** Automatically serializes multi-megabyte weights into `const` C arrays decorated with `PROGMEM` or linker-script specific sections to ensure they remain in Flash ROM and don't overflow the tiny SRAM.

---

## Exhaustive Implementation Checklist

### Phase 1: Core C99 Code Generation Engine & Architecture

- [ ] 1. Implement base C99 AST textual string builder in TS/Python.
- [ ] 2. Define C99 `#include` header generation (`<stdint.h>`, `<math.h>`, `<string.h>`, `<stdbool.h>`).
- [ ] 3. Implement Header Guard generation (`#ifndef MODEL_H ...`).
- [ ] 4. Define the primary `ModelContext` C struct (holding arena pointers and state).
- [ ] 5. Implement `model_init()` C function generator.
- [ ] 6. Implement `model_predict()` / `model_run()` C function generator.
- [ ] 7. Generate explicit function signatures (`int model_predict(const float* input, float* output)`).
- [ ] 8. Auto-format generated C code with a lightweight JS-based C beautifier.
- [ ] 9. Implement `static inline` function generators for small utility math.
- [ ] 10. Support `--namespace` prefixing to avoid C symbol collisions (e.g., `mnist_predict()`).
- [ ] 11. Translate ONNX names to valid C identifiers natively (sanitizing `/`, `-`, `.` to `_`).
- [ ] 12. Ensure generated code compiles warning-free under `gcc -Wall -Wextra -pedantic -std=c99`.
- [ ] 13. Ensure generated code compiles warning-free under `clang`.
- [ ] 14. Support C++ mode (`--emit-cpp`) changing `struct` initialization to support modern C++ standards if requested.
- [ ] 15. Generate `tensor_info` struct for introspecting shapes at runtime without parsing logic.
- [ ] 16. Provide standalone `Makefile` or `CMakeLists.txt` generation alongside the `.c` files.
- [ ] 17. Support generating a standalone `main.c` wrapper for quick local CLI testing.
- [ ] 18. Support explicit pointer restrict qualifiers (`float* restrict out`) for compiler optimization.
- [ ] 19. Embed original ONNX model version and producer metadata as C comments.
- [ ] 20. Track C-stack depth to avoid stack overflows on deeply nested subgraphs.

### Phase 2: Static Memory Arena Allocation (Zero-malloc)

- [ ] 21. Implement static memory planner pass directly on the ONNX AST.
- [ ] 22. Calculate peak working memory required for intermediate activations (the "Arena").
- [ ] 23. Generate a single global contiguous byte array: `static uint8_t arena[ARENA_SIZE];`.
- [ ] 24. Alternatively, allow the user to pass a pre-allocated arena buffer to `model_run(uint8_t* arena)`.
- [ ] 25. Calculate explicit byte offsets for every intermediate tensor (e.g., `float* conv_out = (float*)(&arena[1024]);`).
- [ ] 26. Optimize arena memory by reusing buffer offsets across disjoint topological layers (graph coloring).
- [ ] 27. Ensure strictly aligned memory offsets (4-byte alignment for `float`, 8-byte for `double`).
- [ ] 28. Ensure 16-byte or 32-byte alignment if the user specifies SIMD target compilation.
- [ ] 29. Serialize `Constant` weights into a separate `static const float weights[] = {...}` array.
- [ ] 30. Apply `PROGMEM` macros automatically to constant weight arrays if `--target=arduino` is specified.
- [ ] 31. Apply `__attribute__((section(".rodata")))` for specific bare-metal linker scripts.
- [ ] 32. Split massive weight arrays into smaller chunked C arrays to avoid MSVC/GCC array-size compilation limits.
- [ ] 33. Store quantized (INT8/UINT8) weights as byte arrays to compress the C source file size.
- [ ] 34. Handle explicit C struct alignments (`__attribute__((packed))`) if defining complex tensor headers.
- [ ] 35. Prevent any usage of `malloc`, `calloc`, `realloc`, or `free` in the emitted code.
- [ ] 36. Warn user dynamically if required arena size exceeds standard microcontroller limits (e.g., > 256KB).

### Phase 3: Basic Math & Elementwise Operations (C Loops)

- [ ] 37. Emit C loop for `Add`.
- [ ] 38. Emit C loop for `Sub`.
- [ ] 39. Emit C loop for `Mul`.
- [ ] 40. Emit C loop for `Div`.
- [ ] 41. Handle implicit scalar-to-tensor broadcasting natively within the generated C loop (optimizing out full array iterations).
- [ ] 42. Handle implicit tensor-to-tensor broadcasting (using dynamic stride modulo math in C).
- [ ] 43. Pre-calculate unrolled broadcast offsets statically if dimensions are known, avoiding modulo `%` in C.
- [ ] 44. Emit C `<math.h>` calls for `Exp` (`expf`).
- [ ] 45. Emit C calls for `Log` (`logf`).
- [ ] 46. Emit C calls for `Sqrt` (`sqrtf`).
- [ ] 47. Emit C calls for `Pow` (`powf`).
- [ ] 48. Emit C calls for `Sin` (`sinf`).
- [ ] 49. Emit C calls for `Cos` (`cosf`).
- [ ] 50. Emit C calls for `Tan` (`tanf`).
- [ ] 51. Emit C implementation for `Abs` (`fabsf`).
- [ ] 52. Emit C implementation for `Neg` (`-x`).
- [ ] 53. Emit C implementation for `Sign`.
- [ ] 54. Emit C calls for `Ceil` (`ceilf`).
- [ ] 55. Emit C calls for `Floor` (`floorf`).
- [ ] 56. Emit C calls for `Round` (`roundf`).
- [ ] 57. Implement custom fallback `expf` / `logf` approximation macros if `<math.h>` is forbidden on target.
- [ ] 58. Optimize Division by powers of 2 into right-shifts (`>>`) for integer tensors.
- [ ] 59. Implement strict `NaN` and `Inf` checking macros if `--debug` is enabled.

### Phase 4: Linear Algebra & Matrix Multiplication

- [ ] 60. Emit C loop for `MatMul` (Naive 3-loop structure $O(N^3)$).
- [ ] 61. Emit Cache-Blocked `MatMul` C code (Loop Tiling) for optimized CPU cache utilization.
- [ ] 62. Implement matrix transposition inline within the `MatMul` loop to handle transposed weights.
- [ ] 63. Emit C loop for `Gemm` (Handling `alpha`, `beta`, `transA`, `transB`).
- [ ] 64. Optimize `Gemm` into `MatMul` if `alpha=1.0` and `beta=0.0`.
- [ ] 65. Support batch matrix multiplication (`BatchMatMul`) via 4-loop C structures.
- [ ] 66. Statically unroll small `MatMul` loops (e.g., 3x3 matrices) into linear arithmetic to eliminate loop branching overhead.
- [ ] 67. Detect matrix-vector multiplication (GEMV) and emit specialized, faster $O(N^2)$ loops.
- [ ] 68. Avoid generating inner-loop branches (if statements) during matrix multiplication.
- [ ] 69. Support integer matrix multiplication (`MatMulInteger`) strictly using `int32_t` accumulators to prevent overflow.

### Phase 5: Convolution & Spatial Operations

- [ ] 70. Emit C loop for `Conv` (1D).
- [ ] 71. Emit C loop for `Conv` (2D) natively (7-level deep nested loops for naive direct convolution).
- [ ] 72. Emit C im2col (Image-to-Column) transformation arrays in the arena for optimized `Conv2D`.
- [ ] 73. Map `Conv2D` to im2col + `MatMul` for performance on larger networks.
- [ ] 74. Emit specific C loops for `DepthwiseConv2D` (drastically reducing multiplications).
- [ ] 75. Handle `strides` explicitly within the C loop increments (`i += stride`).
- [ ] 76. Handle `dilations` explicitly within the C loop kernel index lookups.
- [ ] 77. Handle `pads` implicitly by clamping boundary reads (e.g., `if (h < 0 || h >= H) val = 0;`).
- [ ] 78. Alternatively, allocate pre-padded input tensors in the arena to avoid branching in the hot Conv loop.
- [ ] 79. Emit C loop for `ConvTranspose` (2D).
- [ ] 80. Emit 3D Convolution C loops (Video processing).
- [ ] 81. Fuse `Conv` + `BiasAdd` directly into the C accumulation loop to save memory passes.
- [ ] 82. Statically unroll 1x1 Convolutions into direct Matrix Multiplications in C.

### Phase 6: Pooling & Reductions

- [ ] 83. Emit C loop for `MaxPool` (1D, 2D).
- [ ] 84. Emit C loop for `AveragePool` (2D).
- [ ] 85. Handle `kernel_shape`, `strides`, and `pads` dynamically inside pooling loops.
- [ ] 86. Emit C loop for `GlobalAveragePool` (averaging over all spatial dimensions).
- [ ] 87. Emit C loop for `GlobalMaxPool`.
- [ ] 88. Emit C loop for `ReduceMean`.
- [ ] 89. Emit C loop for `ReduceSum`.
- [ ] 90. Emit C loop for `ReduceMax`.
- [ ] 91. Emit C loop for `ReduceMin`.
- [ ] 92. Emit C loop for `ReduceProd`.
- [ ] 93. Handle `keepdims` attribute explicitly in pointer arithmetic mappings.
- [ ] 94. Optimize `ReduceMean` across full continuous flat arrays if spatial dimensions are contiguous.
- [ ] 95. Emit C loop for `ArgMax` (storing the index of the highest value).
- [ ] 96. Emit C loop for `ArgMin`.
- [ ] 97. Provide specific integer precision mappings for `ArgMax` (using `int32_t` or `int64_t`).

### Phase 7: Activations & Normalizations

- [ ] 98. Emit C code for `Relu` (`x > 0 ? x : 0`).
- [ ] 99. Emit C code for `LeakyRelu` (`x > 0 ? x : alpha * x`).
- [ ] 100. Emit C code for `Sigmoid` (`1.0f / (1.0f + expf(-x))`).
- [ ] 101. Emit C code for `Tanh` (`tanhf(x)`).
- [ ] 102. Emit C code for `Softmax`.
- [ ] 103. Implement Numerically Stable Softmax (subtracting max value before exponentiation).
- [ ] 104. Emit C code for `LogSoftmax`.
- [ ] 105. Emit C code for `Gelu` (Erf approximation).
- [ ] 106. Emit C code for `Gelu` (Tanh approximation).
- [ ] 107. Emit C code for `HardSigmoid`.
- [ ] 108. Emit C code for `HardSwish`.
- [ ] 109. Emit C code for `PRelu`.
- [ ] 110. Emit C code for `Clip` (Clamp).
- [ ] 111. Emit C code for `BatchNormalization`.
- [ ] 112. Perform AST constant-folding: Pre-calculate BatchNorm scale and bias offline to eliminate the BatchNorm C loop entirely.
- [ ] 113. Emit C code for `InstanceNormalization`.
- [ ] 114. Emit C code for `LayerNormalization`.
- [ ] 115. Manage LayerNorm `epsilon` precision safely within C types.

### Phase 8: Tensor Manipulation, Shape, & Routing

- [ ] 116. Emit C code for `Reshape` (No-op memory mapping, purely updating C pointer shapes).
- [ ] 117. Emit C code for `Flatten` (No-op).
- [ ] 118. Emit C code for `Squeeze` (No-op).
- [ ] 119. Emit C code for `Unsqueeze` (No-op).
- [ ] 120. Emit C code for `Transpose` (Physical memory copy loop with dimension swapping).
- [ ] 121. Optimize `Transpose` by hoisting standard dimension shifts directly into downstream operator lookup math (virtual transpose).
- [ ] 122. Emit C code for `Concat`.
- [ ] 123. Emit C code for `Split`.
- [ ] 124. Emit C code for `Slice` (Handling static `starts`, `ends`, `axes`, `steps`).
- [ ] 125. Handle dynamic C loops for `Slice` if parameters are runtime variables.
- [ ] 126. Emit C code for `Gather`.
- [ ] 127. Handle Out-Of-Bounds (OOB) indexing in `Gather` safely by clamping to 0 or Max to prevent C Segfaults.
- [ ] 128. Emit C code for `GatherND`.
- [ ] 129. Emit C code for `ScatterElements`.
- [ ] 130. Emit C code for `ScatterND`.
- [ ] 131. Emit C code for `Expand` (Broadcasting tensor to shape).
- [ ] 132. Emit C code for `Tile` (Repeating elements in memory).
- [ ] 133. Emit C code for `Pad` (Generating padded arrays physically in the arena).
- [ ] 134. Emit C code for `ConstantOfShape` (Using `memset` or array fill loop).

### Phase 9: Logical, Relational & Boolean Operations

- [ ] 135. Emit C code for `Equal` (`==`).
- [ ] 136. Emit C code for `Less` (`<`).
- [ ] 137. Emit C code for `LessOrEqual` (`<=`).
- [ ] 138. Emit C code for `Greater` (`>`).
- [ ] 139. Emit C code for `GreaterOrEqual` (`>=`).
- [ ] 140. Emit C code for `Not` (`!`).
- [ ] 141. Emit C code for `And` (`&&`).
- [ ] 142. Emit C code for `Or` (`||`).
- [ ] 143. Emit C code for `Xor` (`!=` for bools).
- [ ] 144. Emit C code for `Where` (Ternary `cond ? x : y`).
- [ ] 145. Handle C type enforcement (mapping ONNX booleans to C99 `bool` or `uint8_t`).

### Phase 10: TinyML Specific Quantization (INT8 / UINT8 / FP16)

- [ ] 146. Emit C loops for `QuantizeLinear`.
- [ ] 147. Emit C loops for `DequantizeLinear`.
- [ ] 148. Emit specific C loops for `QLinearConv` (fully quantized INT8 convolution).
- [ ] 149. Implement precision-safe INT32 accumulators for INT8 `QLinearConv` multiplication.
- [ ] 150. Implement output re-quantization shifts (calculating `M = S1 * S2 / S3` offline and compiling as bit shifts/multipliers).
- [ ] 151. Emit specific C loops for `QLinearMatMul`.
- [ ] 152. Emit specific C loops for `QLinearAdd`.
- [ ] 153. Implement asymmetric zero-point offsets in the C math loops.
- [ ] 154. Pre-pack INT4 (W4A16) weights into `uint8_t` arrays and emit specialized C bit-shifting unpack loops inline.
- [ ] 155. Handle Float16 data types using `_Float16` (C11) or `uint16_t` with software conversion macros.
- [ ] 156. Translate Float16 ops back to Float32 on the fly if the hardware lacks a native FPU for Float16.

### Phase 11: Control Flow & Subgraph Translation

- [ ] 157. Map ONNX `If` to C `if / else` blocks.
- [ ] 158. Generate nested function calls or inline code for `If` branch subgraphs.
- [ ] 159. Map ONNX `Loop` to C `while` or `for` loops.
- [ ] 160. Maintain loop state variables dynamically across C variable scopes.
- [ ] 161. Unroll short loops statically in the AST before C code generation.
- [ ] 162. Ensure Arena planner accounts for memory used within specific subgraphs accurately without overriding parent state.
- [ ] 163. Map ONNX `Scan` operation to a strict C for-loop.

### Phase 12: Hardware-Specific Intrinsics & Pragmas

- [ ] 164. Support `--target=cmsis-nn` to inject ARM Cortex-M DSP intrinsics.
- [ ] 165. Emit `arm_fully_connected_s8` for `QLinearMatMul` if CMSIS-NN is active.
- [ ] 166. Emit `arm_convolve_s8` for `QLinearConv` if CMSIS-NN is active.
- [ ] 167. Support `--target=esp-nn` to inject Espressif ESP32 hardware accelerations.
- [ ] 168. Support `--target=riscv-v` to inject RISC-V Vector extension intrinsics.
- [ ] 169. Emit OpenMP `#pragma omp parallel for` for desktop C generation mode.
- [ ] 170. Generate SIMD loop unrolling hints (`#pragma GCC unroll 4`).
- [ ] 171. Add `#ifdef __AVX2__` headers and block variations for x86 inference.

### Phase 13: Specialized Tasks, NLP & Vision

- [ ] 172. Emit C arrays for BPE Tokenizer dictionaries (enabling text-to-tensor completely in C).
- [ ] 173. Compile ONNX `NonMaxSuppression` (NMS) into a robust C algorithm with dynamic array bounds.
- [ ] 174. Emit C code for `TopK` using a lightweight Quicksort or Heap implementation.
- [ ] 175. Emit C code for `Unique`.
- [ ] 176. Emit C code for `Resize` (Bilinear and Nearest interpolation).
- [ ] 177. Compile `LSTM` natively into a stateful C struct holding hidden states across invocations.
- [ ] 178. Compile `GRU` natively.
- [ ] 179. Compile `RNN` natively.
- [ ] 180. Translate standard PyTorch Attention into explicit C memory loop combinations.

### Phase 14: CLI, Output Formatting & Packaging

- [ ] 181. Build Node.js CLI: `onnx9000 onnx2c model.onnx --output src/`.
- [ ] 182. Support generating a split architecture: `model.c`, `model.h`, and `weights.h`.
- [ ] 183. Bundle external weight binaries into `.bin` files loaded via `fopen` if specifically requested (for Raspberry Pi / Desktop).
- [ ] 184. Implement TS/JS based Zip generator to bundle all C files instantly in the browser.
- [ ] 185. Print memory usage summary table as a C block comment at the top of the header file.
- [ ] 186. Ensure generated names (`model_predict`) can be prefixed dynamically (`--prefix mynet_`).
- [ ] 187. Generate a `.ino` Arduino Sketch file template alongside the C library for instantaneous Arduino IDE compilation.
- [ ] 188. Generate a `CMakeLists.txt` for ESP-IDF integration.

### Phase 15: Edge Cases, Security & Memory Guards

- [ ] 189. Add buffer-overflow protections: Generate `assert()` statements checking tensor array bounds in debug mode.
- [ ] 190. Handle dynamic shapes (`-1` dimensions) by emitting dynamic array variables (`int dim_0 = ...`) rather than `const int` inside the C function.
- [ ] 191. Fallback dynamic arena sizing to stack-allocated VLA (Variable Length Arrays) if C99 VLA is supported and safe.
- [ ] 192. Validate that division by zero is protected via epsilon additions in the C math generation.
- [ ] 193. Ensure C variable scoping doesn't clash (e.g. `int i` reused in nested loops, generate `i_1`, `i_2`).
- [ ] 194. Support `Einsum` by unrolling the string equation directly into nested C loops.
- [ ] 195. Remove any dependencies on `<stdlib.h>` if strictly constrained by the user.
- [ ] 196. Translate `String` type ONNX tensors into `const char*[]` arrays natively.

### Phase 16: Browser UI (The C99 Web Compiler)

- [ ] 197. Build static React/Vue Web UI for `onnx2c`.
- [ ] 198. Implement drag-and-drop ONNX file ingestion.
- [ ] 199. Display an interactive Monaco Editor showing the real-time C99 code generation.
- [ ] 200. Allow the user to tweak memory arena size parameters visually.
- [ ] 201. Support visual toggles for "Enable CMSIS-NN", "Quantize weights to INT8", "Unroll loops".
- [ ] 202. Execute code generation entirely inside a Web Worker.
- [ ] 203. Stream the massive generated text directly into a Blob for local download to prevent browser OOM.
- [ ] 204. Validate model RAM suitability against a dropdown list of standard boards (e.g., "Arduino Nano 33 BLE", "ESP32-S3").

### Phase 17: Validation & Continuous Integration

- [ ] 205. Unit Test: Compile pure `Add` ONNX, pipe through GCC, validate output.
- [ ] 206. Unit Test: Compile `MatMul` ONNX, pipe through GCC, validate output.
- [ ] 207. Unit Test: Compile `Conv2D` ONNX, pipe through GCC, validate output.
- [ ] 208. Integration Test: Convert MNIST CNN, compile via GCC, predict digit successfully in C.
- [ ] 209. Integration Test: Convert MobileNetV2, compile via GCC, profile execution match against Python ORT.
- [ ] 210. Validate generated C syntax strictly against `cppcheck`.
- [ ] 211. Automate generation and compilation of 100+ standard ONNX models in CI testing environments.
- [ ] 212. Verify memory limits on generated binaries using `size` and `objdump`.

### Phase 18: Standard Optimization Passes

- [ ] 213. Execute `onnx9000.optimum` Level 3 graph optimization automatically before C generation.
- [ ] 214. Strip all debug symbols and doc_strings internally before generating the C arrays.
- [ ] 215. Find continuous execution sequences that can be combined (e.g., merging 3 consecutive `Add` operations into `out = x + y + z`).
- [ ] 216. Eliminate transpose operations globally using advanced layout resolution passes.
- [ ] 217. Identify sub-normal float ranges and clamp to zero during generation to prevent extreme CPU penalties in C.

### Phase 19: Extreme TinyML Specializations

- [ ] 218. Provide a LUT (Look Up Table) generator. Compile complex nonlinear functions (e.g. `Sigmoid`) into static pre-computed C arrays `const float sigmoid_lut[256]` to save CPU cycles.
- [ ] 219. Emit bit-packed arrays for boolean tensors (`uint8_t` packing 8 booleans) with masking logic.
- [ ] 220. Support multi-model multiplexing (generating multiple ONNX graphs that share the exact same memory arena).
- [ ] 221. Strip the standard C standard library `math.h` completely if generating integer-only quantized topologies.

### Phase 20: Parity & Delivery Finalization

- [ ] 222. Support compiling specific ONNX `ai.onnx.ml` Classical ML models (Tree Ensembles) to C if-else statements natively.
- [ ] 223. Output `model.c` size must be strictly proportional to the number of nodes (preventing exponential string bloating).
- [ ] 224. Format extremely large floating-point arrays cleanly (`1.234e-5f, ...`) with controlled wrap-around to keep file lines manageable.
- [ ] 225. Release complete API documentation demonstrating how to link the generated `model.h` into a proprietary IoT C codebase.
- [ ] 226. Ensure strict C89 / C99 compliance across all emitted artifacts.
- [ ] 227. Enable users to inject custom C code snippets via attributes (Custom Operator mapping to a manual C function).
- [ ] 228. Provide deterministic C code generation (same ONNX = byte-for-byte identical `.c` output).
- [ ] 229. Ensure no memory leaks exist within the static arena planner.
- [ ] 230. Guarantee absolute standalone execution: compiling the output file with `gcc model.c -lm` works instantly.
- [ ] 231. Handle `float64` fallback cleanly (downcasting to `float32`).
- [ ] 232. Parse specific custom quantization configurations.
- [ ] 233. Generate human-readable tensor names where possible in the C code comments to map back to original ONNX tools.
- [ ] 234. Map `DepthToSpace` using explicit index arithmetic.
- [ ] 235. Map `SpaceToDepth` using explicit index arithmetic.
- [ ] 236. Manage dynamic dimension arrays (`shape = {batch, 3, 224, 224}`) securely in C function signatures.
- [ ] 237. Evaluate static variables completely to `const int` declarations in C.
- [ ] 238. Compile `CumSum` natively.
- [ ] 239. Compile `ReverseSequence` natively.
- [ ] 240. Validate multi-dimensional `GatherND` loops.
- [ ] 241. Validate `ScatterND` memory updates.
- [ ] 242. Translate `OneHot` logic correctly into memset/scatter loops.
- [ ] 243. Add specific macros `ONNX9000_MAX`, `ONNX9000_MIN` to abstract away nested conditionals.
- [ ] 244. Implement memory overlap checking at generation time (ensuring in-place operations don't corrupt dependencies).
- [ ] 245. Render `ConstantOfShape` natively as a C loop setting array bounds dynamically.
- [ ] 246. Compile `NonMaxSuppression` specifically into a static size C array with a returned output count.
- [ ] 247. Provide `#ifdef` toggle to switch between `float` and `double` globally in the generated code.
- [ ] 248. Provide static performance metrics inline (e.g. `// Estimated MACs: 1.2M`).
- [ ] 249. Extract and parse `bfloat16` to `float32` automatically during array generation.
- [ ] 250. Create an interactive AST to C visualization showing which ONNX nodes produced which C functions.
- [ ] 251. Handle Python `tuple` outputs properly via struct return types or multiple pointer arguments in C.
- [ ] 252. Map sequence types to arrays of pointers and dynamic lengths in C.
- [ ] 253. Compile `SequenceConstruct` natively.
- [ ] 254. Compile `SplitToSequence` natively.
- [ ] 255. Provide `__attribute__((aligned(X)))` controls.
- [ ] 256. Handle `tf.js` specific graph structures transpiled into C.
- [ ] 257. Extract strings as `const char*` directly.
- [ ] 258. Support 0-dimension (scalar) tensors implicitly as standard C values rather than pointers.
- [ ] 259. Compile `Pad` values accurately.
- [ ] 260. Output the memory planner allocations dynamically in console logs.
- [ ] 261. Expose the AST compiler via an isolated NPM module `@onnx9000/c-compiler`.
- [ ] 262. Write specific tests for Apple Silicon (M-series) native compilation of the output.
- [ ] 263. Establish a standard interface for custom operator headers.
- [ ] 264. Compile `Erf` into Taylor series approximations if strictly requested.
- [ ] 265. Ensure deterministic float formatting (`%g` or exact hexadecimal IEEE 754 representations in C to avoid precision loss during string translation).
- [ ] 266. Provide array compression (storing repeating patterns via RLE inside the C array, decoding on init).
- [ ] 267. Handle exact INT64 overflow protections statically.
- [ ] 268. Extract 1D vectors seamlessly.
- [ ] 269. Render multidimensional indices properly mapped to flat C arrays.
- [ ] 270. Add support for creating an RTOS-friendly (FreeRTOS) thread-aware task executor macro wrapper.
- [ ] 271. Implement specific memory layouts for HWC image buffering natively.
- [ ] 272. Add dynamic loading `model_load_weights()` to avoid baking multi-megabyte arrays directly into the binary segment.
- [ ] 273. Establish specific error boundaries for missing input pointers.
- [ ] 274. Create UI hooks for importing multiple models into the same C project.
- [ ] 275. Render graph connections in C source comments explicitly.
- [ ] 276. Add flags to limit output verbosity.
- [ ] 277. Automate `npm publish` workflows.
- [ ] 278. Validate the generated code against `MISRA-C` compliance rules for automotive/aerospace reliability.
- [ ] 279. Compile specific scaling factors accurately.
- [ ] 280. Map `Softplus` correctly.
- [ ] 281. Compile `Einsum` cleanly.
- [ ] 282. Add specific code block mapping.
- [ ] 283. Support generating an `.o` object file directly via a WASM-compiled Clang instance in the browser if possible.
- [ ] 284. Allow user configuration of default code spacing and brace styles.
- [ ] 285. Develop detailed JSON output metadata for the generated API.
- [ ] 286. Handle custom ONNX domains explicitly.
- [ ] 287. Implement testing harnesses for Arduino.
- [ ] 288. Emulate unsupported ops cleanly.
- [ ] 289. Map Python `__call__` explicitly.
- [ ] 290. Provide specific configurations for Edge Devices.
- [ ] 291. Build interactive C AST viewers.
- [ ] 292. Add custom Web Workers explicitly to the C code generation string building.
- [ ] 293. Verify all code paths are explicitly typed.
- [ ] 294. Catch `OutOfBounds` array writes during the static planner phase.
- [ ] 295. Configure explicit fallback logic for missing CMSIS-NN branches.
- [ ] 296. Validate TFLite converted models cleanly transpiled.
- [ ] 297. Support conversion directly from `onnx9000.keras`.
- [ ] 298. Validate precise execution under explicit memory bounds checking on real ESP32 silicon.
- [ ] 299. Write comprehensive API documentation mapping C generation targets.
- [ ] 300. Release v1.0 feature complete certification for `onnx9000.onnx2c` achieving full parity with `deepC`.
