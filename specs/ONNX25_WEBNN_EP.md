# ONNX25: WebNN API (Native Browser NPU Execution)

## Original Project Description

The ONNX Runtime WebNN Execution Provider (EP) allows web applications to run ONNX models with hardware acceleration utilizing the emerging W3C Web Neural Network API (WebNN). WebNN provides standard low-level browser APIs to access dedicated machine learning accelerators like Neural Processing Units (NPUs), Digital Signal Processors (DSPs), and specialized GPU ML cores (like Apple's Neural Engine or Intel's VPU/NPU). In the standard ORT architecture, the WebNN EP acts as a bridge: compiling C++ ONNX nodes into JavaScript `MLGraphBuilder` calls via WebAssembly interop.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

`onnx9000` eliminates the C++ WebAssembly middleware entirely for graph construction. Since `onnx9000` handles the ONNX graph directly in TypeScript/JavaScript, the mapping to WebNN is direct, native, and synchronous.

- **Zero JS-WASM Boundary Crossing for Compilation:** The `onnx9000` graph compiler traverses the IR in memory and calls `MLGraphBuilder` natively, compiling the NPU graph orders of magnitude faster than a C++ runtime shuttling strings and pointers across the WASM bridge.
- **Granular Sub-Graph Partitioning:** If the host's NPU/WebNN implementation doesn't support a specific ONNX operator (e.g., a custom transformer node), `onnx9000` dynamically partitions the graph. The supported sub-graphs run natively on the NPU via WebNN, while unsupported nodes seamlessly fall back to `onnx9000`'s highly optimized WebGPU or WASM SIMD backends sharing the same memory context.
- **WebNN Polyfill Integration:** Automatically integrates with the WebNN Polyfill for rapid testing on browsers that have not yet fully shipped the W3C spec.
- **First-Class FP16 & INT8:** WebNN is primarily designed for low-power edge NPUs; `onnx9000` strictly maps its Web-Native W4A16 and INT8 quantizations directly to WebNN primitives to maximize NPU throughput.

---

## Exhaustive Implementation Checklist

### Phase 1: Context Initialization & Feature Detection

- [x] 1. Implement `navigator.ml` presence detection.
- [x] 2. Implement graceful fallback if WebNN API is missing.
- [x] 3. Request `MLContext` via `navigator.ml.createContext()`.
- [x] 4. Support `deviceType: 'npu'` preference.
- [x] 5. Support `deviceType: 'gpu'` preference.
- [x] 6. Support `deviceType: 'cpu'` preference.
- [x] 7. Support `powerPreference: 'default'`.
- [x] 8. Support `powerPreference: 'high-performance'`.
- [x] 9. Support `powerPreference: 'low-power'`.
- [x] 10. Implement caching of the `MLContext` singleton.
- [x] 11. Detect supported data types (`float32`, `float16`, `int32`, `int8`, `uint8`).
- [x] 12. Implement capability queries to check if specific ops are supported by the host context.
- [x] 13. Provide a diagnostic CLI command: `onnx9000 info webnn` to list host NPU capabilities.
- [x] 14. Handle context loss/restore events dynamically.
- [x] 15. Support initializing `MLGraphBuilder` strictly bound to the active context.

### Phase 2: Graph Builder (MLGraphBuilder) Core Orchestration

- [x] 16. Initialize the `MLGraphBuilder` instance.
- [x] 17. Define an internal map of ONNX Node IDs to `MLOperand` objects.
- [x] 18. Implement translation of ONNX Graph Inputs to `builder.input(name, type)`.
- [x] 19. Implement translation of ONNX Initializers to `builder.constant(data)`.
- [x] 20. Resolve ONNX dimensions (Array of Numbers) to WebNN dimensions.
- [x] 21. Map `onnx9000` Float32 tensors to WebNN `float32` constants.
- [x] 22. Map `onnx9000` Float16 tensors to WebNN `float16` constants.
- [x] 23. Map `onnx9000` Int32 tensors to WebNN `int32` constants.
- [x] 24. Map `onnx9000` Int8 tensors to WebNN `int8` constants.
- [x] 25. Map `onnx9000` UInt8 tensors to WebNN `uint8` constants.
- [x] 26. Handle dynamic axes in inputs (specifying `-1` or large bounds if required by specific WebNN drafts).
- [x] 27. Track intermediate `MLOperand` instances during the topological traversal.
- [x] 28. Support releasing intermediate `MLOperand` references to aid garbage collection.
- [x] 29. Map ONNX Graph Outputs to final `MLOperand` evaluations.
- [x] 30. Handle cases where an initializer is passed directly as a graph output.

### Phase 3: Unary & Binary Arithmetic Operations

- [x] 31. Map ONNX `Add` to WebNN `builder.add()`.
- [x] 32. Map ONNX `Sub` to WebNN `builder.sub()`.
- [x] 33. Map ONNX `Mul` to WebNN `builder.mul()`.
- [x] 34. Map ONNX `Div` to WebNN `builder.div()`.
- [x] 35. Map ONNX `Max` to WebNN `builder.max()`.
- [x] 36. Map ONNX `Min` to WebNN `builder.min()`.
- [x] 37. Map ONNX `Pow` to WebNN `builder.pow()`.
- [x] 38. Map ONNX `Abs` to WebNN `builder.abs()`.
- [x] 39. Map ONNX `Ceil` to WebNN `builder.ceil()`.
- [x] 40. Map ONNX `Floor` to WebNN `builder.floor()`.
- [x] 41. Map ONNX `Exp` to WebNN `builder.exp()`.
- [x] 42. Map ONNX `Log` to WebNN `builder.log()`.
- [x] 43. Map ONNX `Cos` to WebNN `builder.cos()`.
- [x] 44. Map ONNX `Sin` to WebNN `builder.sin()`.
- [x] 45. Map ONNX `Tan` to WebNN `builder.tan()`.
- [x] 46. Map ONNX `Acos` to WebNN `builder.acos()`.
- [x] 47. Map ONNX `Asin` to WebNN `builder.asin()`.
- [x] 48. Map ONNX `Atan` to WebNN `builder.atan()`.
- [x] 49. Map ONNX `Sqrt` to WebNN `builder.sqrt()`.
- [x] 50. Map ONNX `Erf` to WebNN `builder.erf()`.
- [x] 51. Map ONNX `Sign` to WebNN `builder.sign()`.
- [x] 52. Map ONNX `Neg` to WebNN `builder.neg()`.
- [x] 53. Handle Numpy-style implicit broadcasting in WebNN binary ops automatically.
- [x] 54. Explicitly reshape scalar initializers for WebNN if the spec requires strict rank matching.

### Phase 4: Activation Functions

- [x] 55. Map ONNX `Relu` to WebNN `builder.relu()`.
- [x] 56. Map ONNX `Sigmoid` to WebNN `builder.sigmoid()`.
- [x] 57. Map ONNX `Tanh` to WebNN `builder.tanh()`.
- [x] 58. Map ONNX `Softmax` to WebNN `builder.softmax()`.
- [x] 59. Handle `Softmax` axis parameter mapping.
- [x] 60. Map ONNX `LeakyRelu` to WebNN `builder.leakyRelu()`.
- [x] 61. Parse `alpha` parameter for `LeakyRelu`.
- [x] 62. Map ONNX `Elu` to WebNN `builder.elu()`.
- [x] 63. Parse `alpha` parameter for `Elu`.
- [x] 64. Map ONNX `HardSigmoid` to WebNN `builder.hardSigmoid()`.
- [x] 65. Parse `alpha` and `beta` parameters for `HardSigmoid`.
- [x] 66. Map ONNX `Softplus` to WebNN `builder.softplus()`.
- [x] 67. Map ONNX `Softsign` to WebNN `builder.softsign()`.
- [x] 68. Map ONNX `Gelu` to WebNN `builder.gelu()`.
- [x] 69. Map ONNX `PRelu` to WebNN `builder.prelu()`.
- [x] 70. Support `Clip` via WebNN `builder.clamp()`.
- [x] 71. Handle missing min/max boundaries in `Clip` converting to infinity bounds.

### Phase 5: Matrix Multiplication & Linear Algebra

- [x] 72. Map ONNX `MatMul` to WebNN `builder.matmul()`.
- [x] 73. Map ONNX `Gemm` to WebNN `builder.gemm()`.
- [x] 74. Parse and apply `alpha` scalar for `Gemm`.
- [x] 75. Parse and apply `beta` scalar for `Gemm`.
- [x] 76. Handle `transA` flag correctly in `Gemm` via WebNN options.
- [x] 77. Handle `transB` flag correctly in `Gemm` via WebNN options.
- [x] 78. Support explicit bias addition in `Gemm` via `c` operand.
- [x] 79. If WebNN `matmul` doesn't support n-dimensional batching natively, emulate via `reshape` -> `matmul` -> `reshape` if mathematically equivalent.
- [x] 80. Fallback: Emulate `Gemm` with `MatMul` + `Add` if `builder.gemm` is missing on specific hardware implementations.
- [x] 81. Implement 1D matrix multiplication bounds checking according to WebNN spec.

### Phase 6: Tensor Manipulation & Routing

- [x] 82. Map ONNX `Reshape` to WebNN `builder.reshape()`.
- [x] 83. Extract dynamic shape tensor inputs to static shapes if WebNN requires static `reshape` arguments at build time.
- [x] 84. Map ONNX `Transpose` to WebNN `builder.transpose()`.
- [x] 85. Pass explicit `permutation` array to `builder.transpose()`.
- [x] 86. Map ONNX `Slice` to WebNN `builder.slice()`.
- [x] 87. Resolve dynamic ONNX `Slice` starts/ends/axes/steps to static WebNN options.
- [x] 88. Emulate negative `starts` and `ends` indices since WebNN slice may require positive absolute bounds.
- [x] 89. Map ONNX `Concat` to WebNN `builder.concat()`.
- [x] 90. Handle `axis` mapping for `Concat`.
- [x] 91. Map ONNX `Split` to WebNN `builder.split()`.
- [x] 92. Handle equal splitting (scalar `split` argument).
- [x] 93. Handle unequal splitting (array `split` argument).
- [x] 94. Map ONNX `Squeeze` to WebNN `builder.reshape()` (calculating squeezed shape dynamically).
- [x] 95. Map ONNX `Unsqueeze` to WebNN `builder.reshape()` (calculating unsqueezed shape dynamically).
- [x] 96. Map ONNX `Expand` to WebNN `builder.expand()`.
- [x] 97. Map ONNX `Gather` to WebNN `builder.gather()`.
- [x] 98. Handle `axis` parameter for `Gather`.
- [x] 99. Handle dynamic/variable indices in `Gather` if WebNN supports them.
- [x] 100. Map ONNX `Tile` by composing `expand` or `concat` ops if direct `tile` is unavailable.
- [x] 101. Map ONNX `Pad` to WebNN `builder.pad()`.
- [x] 102. Handle `constant` padding mode.
- [x] 103. Handle `reflect` padding mode.
- [x] 104. Handle `edge` padding mode.
- [x] 105. Transform ONNX pad tensor format `[x1_begin, x2_begin... x1_end, x2_end...]` to WebNN format `[ [x1_begin, x1_end], [x2_begin, x2_end]... ]`.
- [x] 106. Handle `Cast` using WebNN `builder.cast()`.
- [x] 107. Map ONNX `Shape` to a static CPU/WASM computation since WebNN expects static graphs.

### Phase 7: Convolution & Pooling (Vision Architectures)

- [x] 108. Map ONNX `Conv` (2D) to WebNN `builder.conv2d()`.
- [x] 109. Extract `strides` attribute.
- [x] 110. Extract `dilations` attribute.
- [x] 111. Extract `group` attribute (support Depthwise Conv2D via WebNN groups).
- [x] 112. Map explicit `pads` attribute to WebNN options.
- [x] 113. Implement `auto_pad="SAME_UPPER"` calculation mapping to explicit pad values.
- [x] 114. Implement `auto_pad="SAME_LOWER"` calculation mapping.
- [x] 115. Implement `auto_pad="VALID"` mapping.
- [x] 116. Support passing bias as `bias` option in `conv2d()`.
- [x] 117. Convert ONNX weights (`[M, C/group, kH, kW]`) to WebNN expected layout if default varies (`oihw`).
- [x] 118. Handle `inputLayout` explicitly (`nchw` vs `nhwc`).
- [x] 119. Handle `filterLayout` explicitly (`oihw`, `hwio`, etc.).
- [x] 120. Map ONNX `ConvTranspose` to WebNN `builder.convTranspose2d()`.
- [x] 121. Extract `output_padding` attribute for `ConvTranspose`.
- [x] 122. Map ONNX `MaxPool` to WebNN `builder.maxPool2d()`.
- [x] 123. Map ONNX `AveragePool` to WebNN `builder.averagePool2d()`.
- [x] 124. Handle `kernel_shape` for pooling operations.
- [x] 125. Handle pooling `pads`.
- [x] 126. Handle pooling `strides`.
- [x] 127. Emulate 1D Convolution (`Conv1D`) via WebNN `conv2d` by unsqueezing height=1.
- [x] 128. Emulate 1D Pooling via WebNN `pool2d` by unsqueezing height=1.
- [x] 129. Implement `GlobalAveragePool` via WebNN `builder.averagePool2d()` matching entire spatial dim.
- [x] 130. Implement `GlobalMaxPool` via WebNN `builder.maxPool2d()` matching entire spatial dim.

### Phase 8: Reduction Operations

- [x] 131. Map ONNX `ReduceMean` to WebNN `builder.reduceMean()`.
- [x] 132. Handle `axes` parsing.
- [x] 133. Handle `keepdims` mapping.
- [x] 134. Map ONNX `ReduceSum` to WebNN `builder.reduceSum()`.
- [x] 135. Map ONNX `ReduceMax` to WebNN `builder.reduceMax()`.
- [x] 136. Map ONNX `ReduceMin` to WebNN `builder.reduceMin()`.
- [x] 137. Map ONNX `ReduceProd` to WebNN `builder.reduceProduct()`.
- [x] 138. Map ONNX `ReduceL1` to WebNN `builder.reduceL1()`.
- [x] 139. Map ONNX `ReduceL2` to WebNN `builder.reduceL2()`.
- [x] 140. Map ONNX `ReduceLogSumExp` to WebNN `builder.reduceLogSumExp()`.
- [x] 141. Emulate `ArgMax` via WebNN `builder.argMax()` (if available) or via WebGPU fallback.
- [x] 142. Emulate `ArgMin` via WebNN `builder.argMin()`.

### Phase 9: Normalization Operations

- [x] 143. Map ONNX `BatchNormalization` to WebNN `builder.batchNormalization()`.
- [x] 144. Pass `scale` operand to WebNN.
- [x] 145. Pass `B` (bias) operand to WebNN.
- [x] 146. Pass `mean` operand to WebNN.
- [x] 147. Pass `var` operand to WebNN.
- [x] 148. Parse `epsilon` attribute.
- [x] 149. Map ONNX `InstanceNormalization` to WebNN `builder.instanceNormalization()`.
- [x] 150. Handle `scale` and `B` parameters for InstanceNorm.
- [x] 151. Map ONNX `LayerNormalization` to WebNN `builder.layerNormalization()`.
- [x] 152. Resolve `axis` parameter dynamically for LayerNorm.
- [x] 153. Handle `scale` and `B` parameters for LayerNorm.
- [x] 154. Support `LpNormalization` via WebNN `builder.l2Normalization()`.

### Phase 10: Logical & Relational Operations

- [x] 155. Map ONNX `Equal` to WebNN `builder.equal()`.
- [x] 156. Map ONNX `Greater` to WebNN `builder.greater()`.
- [x] 157. Map ONNX `GreaterOrEqual` to WebNN `builder.greaterOrEqual()`.
- [x] 158. Map ONNX `Less` to WebNN `builder.lesser()`.
- [x] 159. Map ONNX `LessOrEqual` to WebNN `builder.lesserOrEqual()`.
- [x] 160. Map ONNX `Not` to WebNN `builder.logicalNot()`.
- [x] 161. Map ONNX `And` to WebNN `builder.logicalAnd()`.
- [x] 162. Map ONNX `Or` to WebNN `builder.logicalOr()`.
- [x] 163. Map ONNX `Xor` to WebNN `builder.logicalXor()`.
- [x] 164. Map ONNX `Where` to WebNN `builder.where()`.
- [x] 165. Ensure output boolean masks cast strictly back to ONNX Float/Int types if downstream ops require it.

### Phase 11: Graph Compilation & Execution Engine

- [x] 166. Implement the `build()` sequence: finalizing the WebNN `MLGraph`.
- [x] 167. Call `await builder.build(outputs)` to trigger the host NPU compilation.
- [x] 168. Track compile times and log NPU startup latency.
- [x] 169. Allocate `ArrayBuffer` objects for WebNN graph inputs natively in JS.
- [x] 170. Allocate `ArrayBuffer` objects for WebNN graph outputs.
- [x] 171. Map `onnx9000.Tensor` data to WebNN input buffers via `TypedArray` views.
- [x] 172. Implement `context.compute(graph, inputs, outputs)` execution cycle.
- [x] 173. Support the newer `context.dispatch(graph, ...)` API utilizing WebGPU `GPUBuffer` interoperability.
- [x] 174. Enable Zero-Copy execution by mapping `onnx9000` WebGPU tensors directly into WebNN via `MLTensor`.
- [x] 175. Handle execution synchronization (awaiting the NPU compute Promise).
- [x] 176. Re-map WebNN `ArrayBuffer` outputs back to `onnx9000.Tensor` objects safely.
- [x] 177. Maintain an LRU Cache of compiled `MLGraph` objects for dynamic shapes.
- [x] 178. Handle graph disposal via `graph.destroy()` or GC FinalizationRegistry.
- [x] 179. Gracefully catch and log NPU timeout or out-of-memory errors.
- [x] 180. Implement asynchronous non-blocking inference in Web Workers.

### Phase 12: Sub-Graph Partitioning & Fallback

- [x] 181. Implement a WebNN capability checker (simulating a build to check for supported nodes).
- [x] 182. Implement an AST traversal to identify contiguous blocks of WebNN-supported ops.
- [x] 183. Partition the `onnx9000` graph into "WebNN Regions" and "WASM/WebGPU Regions".
- [x] 184. Generate distinct sub-graphs (`onnx9000.Graph`) for each region.
- [x] 185. Compile WebNN Regions to separate `MLGraph` instances.
- [x] 186. Compile WASM/WebGPU Regions using the standard `onnx9000` runtime.
- [x] 187. Execute regions sequentially, copying outputs from WebNN to WASM and vice-versa.
- [x] 188. Optimize boundary crossings (using WebGPU buffers to avoid CPU roundtrips if supported by both).
- [x] 189. Provide CLI flag `--disable-webnn-fallback` to force strict NPU execution (throwing errors if unsupported).
- [x] 190. Handle dynamic shape propagation correctly across partitioned sub-graphs.

### Phase 13: Transformer & LLM specific Operators (WebNN Draft Extensions)

- [x] 191. Map explicit `Gelu` fusions to `builder.gelu()`.
- [x] 192. Translate ONNX `Attention` or `FlashAttention` into standard WebNN MatMul+Softmax subgraphs if a native WebNN Attention op is unavailable.
- [x] 193. Check for emerging W3C WebNN Draft ops (e.g., `triangular`, `scaledDotProductAttention`).
- [x] 194. Fallback: Decompose `LayerNorm` into `ReduceMean`, `Sub`, `Pow`, `Add`, `Div` if `builder.layerNormalization` fails or lacks spec compliance.
- [x] 195. Emulate `RoPE` using WebNN standard trigonometric (`Cos`/`Sin`) and arithmetic primitives.
- [x] 196. Handle multi-dimensional dynamic KV cache updates. If WebNN forbids dynamic `concat`, execute cache updates in WebGPU/WASM and only run the dense feed-forward blocks in WebNN.
- [x] 197. Support caching pre-compiled NPU transformer blocks.
- [x] 198. Map NLP vocabulary `Gather` operations efficiently (or offload to CPU if NPUs struggle with embedding lookups).
- [x] 199. Compile MoE (Mixture of Experts) routers natively in WebNN if conditional execution (`builder.if`) becomes supported.
- [x] 200. Execute gating logic on CPU and only send the selected expert matrices to WebNN to save bandwidth.

### Phase 14: Quantization (W8A8 & W4A16 Native WebNN integration)

- [x] 201. Support ONNX `QuantizeLinear` via WebNN `builder.quantizeLinear()`.
- [x] 202. Support ONNX `DequantizeLinear` via WebNN `builder.dequantizeLinear()`.
- [x] 203. Map ONNX `DynamicQuantizeLinear` to WebNN if supported, otherwise emulate via `reduceMax/Min` and `quantize`.
- [x] 204. Detect and utilize `int8` data types natively in `builder.conv2d` and `builder.matmul`.
- [x] 205. Support zero-point shifting explicitly in WebNN matrix multiplications.
- [x] 206. Implement INT4 unpacking via WebNN bitwise ops (`builder.bitwiseAnd`, `builder.shiftRightLogical`) if available.
- [x] 207. Emulate INT4 unpacking via `float32` math if bitwise ops are missing on the NPU host.
- [x] 208. Integrate with `onnx9000.optimum` to export models specifically targeting WebNN Int8 topologies.
- [x] 209. Push QDQ (Quantize-Dequantize) pairs down the graph into the WebNN compiler, allowing the NPU to fuse them into native Int8 MAC instructions.
- [x] 210. Validate quantization accuracy against CPU baseline to ensure NPU driver hasn't applied aggressive lossy compression.

### Phase 15: Edge Cases & Quirks Management

- [x] 211. Emulate ONNX `GatherElements` (often missing in NPUs) using WebGPU.
- [x] 212. Emulate ONNX `ScatterND` using WebGPU fallback.
- [x] 213. Emulate ONNX `NonZero` (dynamic output shape) by executing exclusively on CPU/WASM.
- [x] 214. Emulate ONNX `TopK` using WASM fallback.
- [x] 215. Handle differences in `padding` spec between ONNX and WebNN (explicit symmetric vs asymmetric arrays).
- [x] 216. Ensure 64-bit integer inputs (`int64`) are automatically down-casted to `int32`, as WebNN officially drops `int64` support for portability.
- [x] 217. Ensure 64-bit floats (`float64`) are down-casted to `float32`.
- [x] 218. Handle empty tensor evaluations (e.g., shape `[0, 10]`) without crashing the NPU driver.
- [x] 219. Manage NaNs and Infs propagation explicitly according to WebNN standard guidelines.
- [x] 220. Prevent WebNN memory limit exceeded crashes by chunking massive convolutions iteratively.

### Phase 16: Device-Specific Tuning (Intel VPU, Apple Neural Engine, Snapdragon)

- [x] 221. Implement a hardware-sniffing utility checking user-agent/GPU strings.
- [x] 222. Workaround: If Apple Neural Engine, prefer NHWC layout explicit casting before `conv2d` to prevent catastrophic driver reshapes.
- [x] 223. Workaround: If Intel VPU, pad channel dimensions to multiples of 4 or 16.
- [x] 224. Workaround: Avoid `builder.erf()` on Snapdragon NPUs if known to be unstable, emulating with Tanh polynomials.
- [x] 225. Workaround: Detect driver timeouts on Windows ARM and reduce graph partition sizes automatically.
- [x] 226. Provide a `--webnn-compat-mode` flag enabling all known driver workarounds.
- [x] 227. Profile operator compilation times to dynamically skip WebNN for extremely fast, simple nodes (e.g., a single `Add`), which are faster in WASM.
- [x] 228. Support pre-warming the NPU with dummy data to avoid UI stutters on first inference.
- [x] 229. Expose NPU execution metrics natively via the `onnx9000` Profiler API.
- [x] 230. Submit anonymous telemetry on specific WebNN operator failures to identify broken driver updates.

### Phase 17: Memory Management & Buffer Re-use

- [x] 231. Implement an Arena allocator specifically for WebNN `ArrayBuffer` inputs.
- [x] 232. Prevent garbage collection thrashing by re-using `context.compute` output buffers.
- [x] 233. Map `onnx9000` internal tensor pools directly to WebNN view allocations.
- [x] 234. Handle sub-array views cleanly (when an ONNX tensor is merely a sliced view of another memory block).
- [x] 235. Support zero-initialization of padding buffers to prevent security leaks of old memory.
- [x] 236. Manage WebGPU `MLTensor` lifecycles properly, calling `tensor.destroy()` precisely when the graph is destroyed.
- [x] 237. Ensure asynchronous execution prevents memory mutations from the main thread during NPU execution.
- [x] 238. Fallback to copying buffers securely if SharedArrayBuffer is restricted by CORS/COOP headers.
- [x] 239. Monitor JS heap size vs active WebNN allocations, triggering manual GC hints if nearing OOM.
- [x] 240. Track precise byte alignment requirements (e.g., 4-byte boundaries) for `float16` buffers passed to WebNN.

### Phase 18: Testing & Conformance

- [x] 241. Construct automated test suite passing the standard ONNX Node test dataset directly to the WebNN EP.
- [x] 242. Validate `Add` node outputs against WASM CPU.
- [x] 243. Validate `Conv2d` node outputs against WASM CPU.
- [x] 244. Validate `MatMul` node outputs against WASM CPU.
- [x] 245. Run tests using the `webnn-polyfill` in headless Chrome/Puppeteer.
- [x] 246. Run tests natively on macOS Chrome with `--enable-features=WebMachineLearningNeuralNetwork`.
- [x] 247. Run tests natively on Windows Edge with NPU support enabled.
- [x] 248. Calculate acceptable numerical drift tolerances (e.g., 1e-4) to account for NPU-specific precision drops.
- [x] 249. Create tests for every single pad mode (`constant`, `edge`, `reflect`).
- [x] 250. Create tests for specific broadcast combinations (e.g., `[1, 3, 224, 224] + [3, 1, 1]`).
- [x] 251. Test multi-output nodes (e.g., `Split`, `TopK` fallback) correctness.
- [x] 252. Ensure memory is pristine after 1000 successive iterations (leak testing).
- [x] 253. Build a fuzzing harness generating random ONNX graphs and ensuring the WebNN EP doesn't crash the browser.
- [x] 254. Test dynamic batch sizes (changing input shape from `[1, ...]` to `[4, ...]`) without re-compiling the graph.
- [x] 255. Verify asynchronous execution does not block CSS animations on the main thread.

### Phase 19: Framework & Tooling Integration

- [x] 256. Allow `Transformers.js` pipelines to explicitly target WebNN (`device: 'webnn'`).
- [x] 257. Hook WebNN capability checking into the `AutoConfig` loader.
- [x] 258. Ensure `onnx9000.genai` can offload LLM MatMul blocks natively to the NPU.
- [x] 259. Integrate with `onnx9000.optimum` CLI to allow testing WebNN equivalence directly from the command line (`onnx9000 test webnn model.onnx`).
- [x] 260. Publish a diagnostic web page showing "WebNN Readiness" for a user's current browser.
- [x] 261. Integrate with React Native/Expo (when WebNN ships to mobile WebViews).
- [x] 262. Support WebNN EP configuration flags (e.g., setting execution priority).
- [x] 263. Emit standard `onnxruntime` EP log formats for compatibility with legacy debugging tools.
- [x] 264. Support importing generic ONNX JSON (via ORT) and building the WebNN graph.
- [x] 265. Document the complete list of supported ops and their spec version in a generated Markdown file.

### Phase 20: Advanced API Features & Future Specs

- [x] 266. Prepare for W3C WebNN API v2 (dynamic shapes natively).
- [x] 267. Map ONNX `Loop` natively if WebNN introduces control flow APIs.
- [x] 268. Map ONNX `If` natively to WebNN.
- [x] 269. Support specialized WebNN `lstm` and `gru` builder functions for RNN models.
- [x] 270. Support WebNN `builder.resample2d` explicitly for ONNX `Resize` operations.
- [x] 271. Support nearest-neighbor interpolation in WebNN `resample2d`.
- [x] 272. Support linear interpolation in WebNN `resample2d`.
- [x] 273. Support `builder.gatherNd` if added to the WebNN spec.
- [x] 274. Handle WebNN `logicalAnd/Or/Not` applied to multi-dimensional masks.
- [x] 275. Map ONNX `CumSum` to NPU native execution (often tricky, might require scan algorithms).
- [x] 276. Provide hooks for WebNN `builder.gruCell` mapping.
- [x] 277. Provide hooks for WebNN `builder.lstmCell` mapping.
- [x] 278. Support explicit data layout overriding during WebNN graph build (ignoring ONNX constraints).
- [x] 279. Build an automated transpiler: `onnx9000-to-wgsl` for ops rejected by the WebNN context, ensuring no fallback to slow JS math.
- [x] 280. Handle `uint32` data types in WebNN (often required for Gather indices).
- [x] 281. Integrate `onnx9000.image` pre-processing natively into the WebNN graph (fusing Normalize/Resize ops into the NPU).
- [x] 282. Expose `builder.triangular` for specialized causal masking if present.
- [x] 283. Support executing multiple isolated WebNN contexts concurrently for multi-model web apps.
- [x] 284. Implement fallback logic for WebNN unsupported `dilations` values in specific layers.
- [x] 285. Support `builder.dequantizeLinear` executing specifically on NPU vector engines.
- [x] 286. Map ONNX `SpaceToDepth` and `DepthToSpace` to WebNN if supported natively.
- [x] 287. Compile and run YOLO-v8 fully accelerated on the WebNN EP.
- [x] 288. Compile and run MobileViT fully accelerated on the WebNN EP.
- [x] 289. Compile and run Whisper (Encoder) fully accelerated on the WebNN EP.
- [x] 290. Maintain an architecture compatibility matrix tracking exact NPU support levels (Qualcomm vs Apple vs Intel).
- [x] 291. Validate exact compliance with WebNN Draft Spec W3C Working Drafts.
- [x] 292. Support `builder.concat` with more than 5 inputs (handling NPU argument limits).
- [x] 293. Track and bypass known WebNN Polyfill bugs dynamically.
- [x] 294. Optimize constant memory uploads to prevent Chrome UI freezes during `builder.build()`.
- [x] 295. Execute deep layout analysis (NCHW to NHWC) eliminating redundant transpose chains specific to NPU backends.
- [x] 296. Map ONNX `HardSwish` natively using WebNN arithmetic `x * hardSigmoid`.
- [x] 297. Support WebNN native `builder.softplus`.
- [x] 298. Validate precise execution parity between `device: 'webgpu'` and `device: 'webnn'` on the exact same hardware.
- [x] 299. Write comprehensive tutorial: "Deploying ONNX Models to NPUs in the Browser".
- [x] 300. Release v1.0 complete feature parity certification matching the official C++ ONNX Runtime WebNN EP.
