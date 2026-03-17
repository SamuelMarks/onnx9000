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

- [ ] 1. Implement `navigator.ml` presence detection.
- [ ] 2. Implement graceful fallback if WebNN API is missing.
- [ ] 3. Request `MLContext` via `navigator.ml.createContext()`.
- [ ] 4. Support `deviceType: 'npu'` preference.
- [ ] 5. Support `deviceType: 'gpu'` preference.
- [ ] 6. Support `deviceType: 'cpu'` preference.
- [ ] 7. Support `powerPreference: 'default'`.
- [ ] 8. Support `powerPreference: 'high-performance'`.
- [ ] 9. Support `powerPreference: 'low-power'`.
- [ ] 10. Implement caching of the `MLContext` singleton.
- [ ] 11. Detect supported data types (`float32`, `float16`, `int32`, `int8`, `uint8`).
- [ ] 12. Implement capability queries to check if specific ops are supported by the host context.
- [ ] 13. Provide a diagnostic CLI command: `onnx9000 info webnn` to list host NPU capabilities.
- [ ] 14. Handle context loss/restore events dynamically.
- [ ] 15. Support initializing `MLGraphBuilder` strictly bound to the active context.

### Phase 2: Graph Builder (MLGraphBuilder) Core Orchestration

- [ ] 16. Initialize the `MLGraphBuilder` instance.
- [ ] 17. Define an internal map of ONNX Node IDs to `MLOperand` objects.
- [ ] 18. Implement translation of ONNX Graph Inputs to `builder.input(name, type)`.
- [ ] 19. Implement translation of ONNX Initializers to `builder.constant(data)`.
- [ ] 20. Resolve ONNX dimensions (Array of Numbers) to WebNN dimensions.
- [ ] 21. Map `onnx9000` Float32 tensors to WebNN `float32` constants.
- [ ] 22. Map `onnx9000` Float16 tensors to WebNN `float16` constants.
- [ ] 23. Map `onnx9000` Int32 tensors to WebNN `int32` constants.
- [ ] 24. Map `onnx9000` Int8 tensors to WebNN `int8` constants.
- [ ] 25. Map `onnx9000` UInt8 tensors to WebNN `uint8` constants.
- [ ] 26. Handle dynamic axes in inputs (specifying `-1` or large bounds if required by specific WebNN drafts).
- [ ] 27. Track intermediate `MLOperand` instances during the topological traversal.
- [ ] 28. Support releasing intermediate `MLOperand` references to aid garbage collection.
- [ ] 29. Map ONNX Graph Outputs to final `MLOperand` evaluations.
- [ ] 30. Handle cases where an initializer is passed directly as a graph output.

### Phase 3: Unary & Binary Arithmetic Operations

- [ ] 31. Map ONNX `Add` to WebNN `builder.add()`.
- [ ] 32. Map ONNX `Sub` to WebNN `builder.sub()`.
- [ ] 33. Map ONNX `Mul` to WebNN `builder.mul()`.
- [ ] 34. Map ONNX `Div` to WebNN `builder.div()`.
- [ ] 35. Map ONNX `Max` to WebNN `builder.max()`.
- [ ] 36. Map ONNX `Min` to WebNN `builder.min()`.
- [ ] 37. Map ONNX `Pow` to WebNN `builder.pow()`.
- [ ] 38. Map ONNX `Abs` to WebNN `builder.abs()`.
- [ ] 39. Map ONNX `Ceil` to WebNN `builder.ceil()`.
- [ ] 40. Map ONNX `Floor` to WebNN `builder.floor()`.
- [ ] 41. Map ONNX `Exp` to WebNN `builder.exp()`.
- [ ] 42. Map ONNX `Log` to WebNN `builder.log()`.
- [ ] 43. Map ONNX `Cos` to WebNN `builder.cos()`.
- [ ] 44. Map ONNX `Sin` to WebNN `builder.sin()`.
- [ ] 45. Map ONNX `Tan` to WebNN `builder.tan()`.
- [ ] 46. Map ONNX `Acos` to WebNN `builder.acos()`.
- [ ] 47. Map ONNX `Asin` to WebNN `builder.asin()`.
- [ ] 48. Map ONNX `Atan` to WebNN `builder.atan()`.
- [ ] 49. Map ONNX `Sqrt` to WebNN `builder.sqrt()`.
- [ ] 50. Map ONNX `Erf` to WebNN `builder.erf()`.
- [ ] 51. Map ONNX `Sign` to WebNN `builder.sign()`.
- [ ] 52. Map ONNX `Neg` to WebNN `builder.neg()`.
- [ ] 53. Handle Numpy-style implicit broadcasting in WebNN binary ops automatically.
- [ ] 54. Explicitly reshape scalar initializers for WebNN if the spec requires strict rank matching.

### Phase 4: Activation Functions

- [ ] 55. Map ONNX `Relu` to WebNN `builder.relu()`.
- [ ] 56. Map ONNX `Sigmoid` to WebNN `builder.sigmoid()`.
- [ ] 57. Map ONNX `Tanh` to WebNN `builder.tanh()`.
- [ ] 58. Map ONNX `Softmax` to WebNN `builder.softmax()`.
- [ ] 59. Handle `Softmax` axis parameter mapping.
- [ ] 60. Map ONNX `LeakyRelu` to WebNN `builder.leakyRelu()`.
- [ ] 61. Parse `alpha` parameter for `LeakyRelu`.
- [ ] 62. Map ONNX `Elu` to WebNN `builder.elu()`.
- [ ] 63. Parse `alpha` parameter for `Elu`.
- [ ] 64. Map ONNX `HardSigmoid` to WebNN `builder.hardSigmoid()`.
- [ ] 65. Parse `alpha` and `beta` parameters for `HardSigmoid`.
- [ ] 66. Map ONNX `Softplus` to WebNN `builder.softplus()`.
- [ ] 67. Map ONNX `Softsign` to WebNN `builder.softsign()`.
- [ ] 68. Map ONNX `Gelu` to WebNN `builder.gelu()`.
- [ ] 69. Map ONNX `PRelu` to WebNN `builder.prelu()`.
- [ ] 70. Support `Clip` via WebNN `builder.clamp()`.
- [ ] 71. Handle missing min/max boundaries in `Clip` converting to infinity bounds.

### Phase 5: Matrix Multiplication & Linear Algebra

- [ ] 72. Map ONNX `MatMul` to WebNN `builder.matmul()`.
- [ ] 73. Map ONNX `Gemm` to WebNN `builder.gemm()`.
- [ ] 74. Parse and apply `alpha` scalar for `Gemm`.
- [ ] 75. Parse and apply `beta` scalar for `Gemm`.
- [ ] 76. Handle `transA` flag correctly in `Gemm` via WebNN options.
- [ ] 77. Handle `transB` flag correctly in `Gemm` via WebNN options.
- [ ] 78. Support explicit bias addition in `Gemm` via `c` operand.
- [ ] 79. If WebNN `matmul` doesn't support n-dimensional batching natively, emulate via `reshape` -> `matmul` -> `reshape` if mathematically equivalent.
- [ ] 80. Fallback: Emulate `Gemm` with `MatMul` + `Add` if `builder.gemm` is missing on specific hardware implementations.
- [ ] 81. Implement 1D matrix multiplication bounds checking according to WebNN spec.

### Phase 6: Tensor Manipulation & Routing

- [ ] 82. Map ONNX `Reshape` to WebNN `builder.reshape()`.
- [ ] 83. Extract dynamic shape tensor inputs to static shapes if WebNN requires static `reshape` arguments at build time.
- [ ] 84. Map ONNX `Transpose` to WebNN `builder.transpose()`.
- [ ] 85. Pass explicit `permutation` array to `builder.transpose()`.
- [ ] 86. Map ONNX `Slice` to WebNN `builder.slice()`.
- [ ] 87. Resolve dynamic ONNX `Slice` starts/ends/axes/steps to static WebNN options.
- [ ] 88. Emulate negative `starts` and `ends` indices since WebNN slice may require positive absolute bounds.
- [ ] 89. Map ONNX `Concat` to WebNN `builder.concat()`.
- [ ] 90. Handle `axis` mapping for `Concat`.
- [ ] 91. Map ONNX `Split` to WebNN `builder.split()`.
- [ ] 92. Handle equal splitting (scalar `split` argument).
- [ ] 93. Handle unequal splitting (array `split` argument).
- [ ] 94. Map ONNX `Squeeze` to WebNN `builder.reshape()` (calculating squeezed shape dynamically).
- [ ] 95. Map ONNX `Unsqueeze` to WebNN `builder.reshape()` (calculating unsqueezed shape dynamically).
- [ ] 96. Map ONNX `Expand` to WebNN `builder.expand()`.
- [ ] 97. Map ONNX `Gather` to WebNN `builder.gather()`.
- [ ] 98. Handle `axis` parameter for `Gather`.
- [ ] 99. Handle dynamic/variable indices in `Gather` if WebNN supports them.
- [ ] 100. Map ONNX `Tile` by composing `expand` or `concat` ops if direct `tile` is unavailable.
- [ ] 101. Map ONNX `Pad` to WebNN `builder.pad()`.
- [ ] 102. Handle `constant` padding mode.
- [ ] 103. Handle `reflect` padding mode.
- [ ] 104. Handle `edge` padding mode.
- [ ] 105. Transform ONNX pad tensor format `[x1_begin, x2_begin... x1_end, x2_end...]` to WebNN format `[ [x1_begin, x1_end], [x2_begin, x2_end]... ]`.
- [ ] 106. Handle `Cast` using WebNN `builder.cast()`.
- [ ] 107. Map ONNX `Shape` to a static CPU/WASM computation since WebNN expects static graphs.

### Phase 7: Convolution & Pooling (Vision Architectures)

- [ ] 108. Map ONNX `Conv` (2D) to WebNN `builder.conv2d()`.
- [ ] 109. Extract `strides` attribute.
- [ ] 110. Extract `dilations` attribute.
- [ ] 111. Extract `group` attribute (support Depthwise Conv2D via WebNN groups).
- [ ] 112. Map explicit `pads` attribute to WebNN options.
- [ ] 113. Implement `auto_pad="SAME_UPPER"` calculation mapping to explicit pad values.
- [ ] 114. Implement `auto_pad="SAME_LOWER"` calculation mapping.
- [ ] 115. Implement `auto_pad="VALID"` mapping.
- [ ] 116. Support passing bias as `bias` option in `conv2d()`.
- [ ] 117. Convert ONNX weights (`[M, C/group, kH, kW]`) to WebNN expected layout if default varies (`oihw`).
- [ ] 118. Handle `inputLayout` explicitly (`nchw` vs `nhwc`).
- [ ] 119. Handle `filterLayout` explicitly (`oihw`, `hwio`, etc.).
- [ ] 120. Map ONNX `ConvTranspose` to WebNN `builder.convTranspose2d()`.
- [ ] 121. Extract `output_padding` attribute for `ConvTranspose`.
- [ ] 122. Map ONNX `MaxPool` to WebNN `builder.maxPool2d()`.
- [ ] 123. Map ONNX `AveragePool` to WebNN `builder.averagePool2d()`.
- [ ] 124. Handle `kernel_shape` for pooling operations.
- [ ] 125. Handle pooling `pads`.
- [ ] 126. Handle pooling `strides`.
- [ ] 127. Emulate 1D Convolution (`Conv1D`) via WebNN `conv2d` by unsqueezing height=1.
- [ ] 128. Emulate 1D Pooling via WebNN `pool2d` by unsqueezing height=1.
- [ ] 129. Implement `GlobalAveragePool` via WebNN `builder.averagePool2d()` matching entire spatial dim.
- [ ] 130. Implement `GlobalMaxPool` via WebNN `builder.maxPool2d()` matching entire spatial dim.

### Phase 8: Reduction Operations

- [ ] 131. Map ONNX `ReduceMean` to WebNN `builder.reduceMean()`.
- [ ] 132. Handle `axes` parsing.
- [ ] 133. Handle `keepdims` mapping.
- [ ] 134. Map ONNX `ReduceSum` to WebNN `builder.reduceSum()`.
- [ ] 135. Map ONNX `ReduceMax` to WebNN `builder.reduceMax()`.
- [ ] 136. Map ONNX `ReduceMin` to WebNN `builder.reduceMin()`.
- [ ] 137. Map ONNX `ReduceProd` to WebNN `builder.reduceProduct()`.
- [ ] 138. Map ONNX `ReduceL1` to WebNN `builder.reduceL1()`.
- [ ] 139. Map ONNX `ReduceL2` to WebNN `builder.reduceL2()`.
- [ ] 140. Map ONNX `ReduceLogSumExp` to WebNN `builder.reduceLogSumExp()`.
- [ ] 141. Emulate `ArgMax` via WebNN `builder.argMax()` (if available) or via WebGPU fallback.
- [ ] 142. Emulate `ArgMin` via WebNN `builder.argMin()`.

### Phase 9: Normalization Operations

- [ ] 143. Map ONNX `BatchNormalization` to WebNN `builder.batchNormalization()`.
- [ ] 144. Pass `scale` operand to WebNN.
- [ ] 145. Pass `B` (bias) operand to WebNN.
- [ ] 146. Pass `mean` operand to WebNN.
- [ ] 147. Pass `var` operand to WebNN.
- [ ] 148. Parse `epsilon` attribute.
- [ ] 149. Map ONNX `InstanceNormalization` to WebNN `builder.instanceNormalization()`.
- [ ] 150. Handle `scale` and `B` parameters for InstanceNorm.
- [ ] 151. Map ONNX `LayerNormalization` to WebNN `builder.layerNormalization()`.
- [ ] 152. Resolve `axis` parameter dynamically for LayerNorm.
- [ ] 153. Handle `scale` and `B` parameters for LayerNorm.
- [ ] 154. Support `LpNormalization` via WebNN `builder.l2Normalization()`.

### Phase 10: Logical & Relational Operations

- [ ] 155. Map ONNX `Equal` to WebNN `builder.equal()`.
- [ ] 156. Map ONNX `Greater` to WebNN `builder.greater()`.
- [ ] 157. Map ONNX `GreaterOrEqual` to WebNN `builder.greaterOrEqual()`.
- [ ] 158. Map ONNX `Less` to WebNN `builder.lesser()`.
- [ ] 159. Map ONNX `LessOrEqual` to WebNN `builder.lesserOrEqual()`.
- [ ] 160. Map ONNX `Not` to WebNN `builder.logicalNot()`.
- [ ] 161. Map ONNX `And` to WebNN `builder.logicalAnd()`.
- [ ] 162. Map ONNX `Or` to WebNN `builder.logicalOr()`.
- [ ] 163. Map ONNX `Xor` to WebNN `builder.logicalXor()`.
- [ ] 164. Map ONNX `Where` to WebNN `builder.where()`.
- [ ] 165. Ensure output boolean masks cast strictly back to ONNX Float/Int types if downstream ops require it.

### Phase 11: Graph Compilation & Execution Engine

- [ ] 166. Implement the `build()` sequence: finalizing the WebNN `MLGraph`.
- [ ] 167. Call `await builder.build(outputs)` to trigger the host NPU compilation.
- [ ] 168. Track compile times and log NPU startup latency.
- [ ] 169. Allocate `ArrayBuffer` objects for WebNN graph inputs natively in JS.
- [ ] 170. Allocate `ArrayBuffer` objects for WebNN graph outputs.
- [ ] 171. Map `onnx9000.Tensor` data to WebNN input buffers via `TypedArray` views.
- [ ] 172. Implement `context.compute(graph, inputs, outputs)` execution cycle.
- [ ] 173. Support the newer `context.dispatch(graph, ...)` API utilizing WebGPU `GPUBuffer` interoperability.
- [ ] 174. Enable Zero-Copy execution by mapping `onnx9000` WebGPU tensors directly into WebNN via `MLTensor`.
- [ ] 175. Handle execution synchronization (awaiting the NPU compute Promise).
- [ ] 176. Re-map WebNN `ArrayBuffer` outputs back to `onnx9000.Tensor` objects safely.
- [ ] 177. Maintain an LRU Cache of compiled `MLGraph` objects for dynamic shapes.
- [ ] 178. Handle graph disposal via `graph.destroy()` or GC FinalizationRegistry.
- [ ] 179. Gracefully catch and log NPU timeout or out-of-memory errors.
- [ ] 180. Implement asynchronous non-blocking inference in Web Workers.

### Phase 12: Sub-Graph Partitioning & Fallback

- [ ] 181. Implement a WebNN capability checker (simulating a build to check for supported nodes).
- [ ] 182. Implement an AST traversal to identify contiguous blocks of WebNN-supported ops.
- [ ] 183. Partition the `onnx9000` graph into "WebNN Regions" and "WASM/WebGPU Regions".
- [ ] 184. Generate distinct sub-graphs (`onnx9000.Graph`) for each region.
- [ ] 185. Compile WebNN Regions to separate `MLGraph` instances.
- [ ] 186. Compile WASM/WebGPU Regions using the standard `onnx9000` runtime.
- [ ] 187. Execute regions sequentially, copying outputs from WebNN to WASM and vice-versa.
- [ ] 188. Optimize boundary crossings (using WebGPU buffers to avoid CPU roundtrips if supported by both).
- [ ] 189. Provide CLI flag `--disable-webnn-fallback` to force strict NPU execution (throwing errors if unsupported).
- [ ] 190. Handle dynamic shape propagation correctly across partitioned sub-graphs.

### Phase 13: Transformer & LLM specific Operators (WebNN Draft Extensions)

- [ ] 191. Map explicit `Gelu` fusions to `builder.gelu()`.
- [ ] 192. Translate ONNX `Attention` or `FlashAttention` into standard WebNN MatMul+Softmax subgraphs if a native WebNN Attention op is unavailable.
- [ ] 193. Check for emerging W3C WebNN Draft ops (e.g., `triangular`, `scaledDotProductAttention`).
- [ ] 194. Fallback: Decompose `LayerNorm` into `ReduceMean`, `Sub`, `Pow`, `Add`, `Div` if `builder.layerNormalization` fails or lacks spec compliance.
- [ ] 195. Emulate `RoPE` using WebNN standard trigonometric (`Cos`/`Sin`) and arithmetic primitives.
- [ ] 196. Handle multi-dimensional dynamic KV cache updates. If WebNN forbids dynamic `concat`, execute cache updates in WebGPU/WASM and only run the dense feed-forward blocks in WebNN.
- [ ] 197. Support caching pre-compiled NPU transformer blocks.
- [ ] 198. Map NLP vocabulary `Gather` operations efficiently (or offload to CPU if NPUs struggle with embedding lookups).
- [ ] 199. Compile MoE (Mixture of Experts) routers natively in WebNN if conditional execution (`builder.if`) becomes supported.
- [ ] 200. Execute gating logic on CPU and only send the selected expert matrices to WebNN to save bandwidth.

### Phase 14: Quantization (W8A8 & W4A16 Native WebNN integration)

- [ ] 201. Support ONNX `QuantizeLinear` via WebNN `builder.quantizeLinear()`.
- [ ] 202. Support ONNX `DequantizeLinear` via WebNN `builder.dequantizeLinear()`.
- [ ] 203. Map ONNX `DynamicQuantizeLinear` to WebNN if supported, otherwise emulate via `reduceMax/Min` and `quantize`.
- [ ] 204. Detect and utilize `int8` data types natively in `builder.conv2d` and `builder.matmul`.
- [ ] 205. Support zero-point shifting explicitly in WebNN matrix multiplications.
- [ ] 206. Implement INT4 unpacking via WebNN bitwise ops (`builder.bitwiseAnd`, `builder.shiftRightLogical`) if available.
- [ ] 207. Emulate INT4 unpacking via `float32` math if bitwise ops are missing on the NPU host.
- [ ] 208. Integrate with `onnx9000.optimum` to export models specifically targeting WebNN Int8 topologies.
- [ ] 209. Push QDQ (Quantize-Dequantize) pairs down the graph into the WebNN compiler, allowing the NPU to fuse them into native Int8 MAC instructions.
- [ ] 210. Validate quantization accuracy against CPU baseline to ensure NPU driver hasn't applied aggressive lossy compression.

### Phase 15: Edge Cases & Quirks Management

- [ ] 211. Emulate ONNX `GatherElements` (often missing in NPUs) using WebGPU.
- [ ] 212. Emulate ONNX `ScatterND` using WebGPU fallback.
- [ ] 213. Emulate ONNX `NonZero` (dynamic output shape) by executing exclusively on CPU/WASM.
- [ ] 214. Emulate ONNX `TopK` using WASM fallback.
- [ ] 215. Handle differences in `padding` spec between ONNX and WebNN (explicit symmetric vs asymmetric arrays).
- [ ] 216. Ensure 64-bit integer inputs (`int64`) are automatically down-casted to `int32`, as WebNN officially drops `int64` support for portability.
- [ ] 217. Ensure 64-bit floats (`float64`) are down-casted to `float32`.
- [ ] 218. Handle empty tensor evaluations (e.g., shape `[0, 10]`) without crashing the NPU driver.
- [ ] 219. Manage NaNs and Infs propagation explicitly according to WebNN standard guidelines.
- [ ] 220. Prevent WebNN memory limit exceeded crashes by chunking massive convolutions iteratively.

### Phase 16: Device-Specific Tuning (Intel VPU, Apple Neural Engine, Snapdragon)

- [ ] 221. Implement a hardware-sniffing utility checking user-agent/GPU strings.
- [ ] 222. Workaround: If Apple Neural Engine, prefer NHWC layout explicit casting before `conv2d` to prevent catastrophic driver reshapes.
- [ ] 223. Workaround: If Intel VPU, pad channel dimensions to multiples of 4 or 16.
- [ ] 224. Workaround: Avoid `builder.erf()` on Snapdragon NPUs if known to be unstable, emulating with Tanh polynomials.
- [ ] 225. Workaround: Detect driver timeouts on Windows ARM and reduce graph partition sizes automatically.
- [ ] 226. Provide a `--webnn-compat-mode` flag enabling all known driver workarounds.
- [ ] 227. Profile operator compilation times to dynamically skip WebNN for extremely fast, simple nodes (e.g., a single `Add`), which are faster in WASM.
- [ ] 228. Support pre-warming the NPU with dummy data to avoid UI stutters on first inference.
- [ ] 229. Expose NPU execution metrics natively via the `onnx9000` Profiler API.
- [ ] 230. Submit anonymous telemetry on specific WebNN operator failures to identify broken driver updates.

### Phase 17: Memory Management & Buffer Re-use

- [ ] 231. Implement an Arena allocator specifically for WebNN `ArrayBuffer` inputs.
- [ ] 232. Prevent garbage collection thrashing by re-using `context.compute` output buffers.
- [ ] 233. Map `onnx9000` internal tensor pools directly to WebNN view allocations.
- [ ] 234. Handle sub-array views cleanly (when an ONNX tensor is merely a sliced view of another memory block).
- [ ] 235. Support zero-initialization of padding buffers to prevent security leaks of old memory.
- [ ] 236. Manage WebGPU `MLTensor` lifecycles properly, calling `tensor.destroy()` precisely when the graph is destroyed.
- [ ] 237. Ensure asynchronous execution prevents memory mutations from the main thread during NPU execution.
- [ ] 238. Fallback to copying buffers securely if SharedArrayBuffer is restricted by CORS/COOP headers.
- [ ] 239. Monitor JS heap size vs active WebNN allocations, triggering manual GC hints if nearing OOM.
- [ ] 240. Track precise byte alignment requirements (e.g., 4-byte boundaries) for `float16` buffers passed to WebNN.

### Phase 18: Testing & Conformance

- [ ] 241. Construct automated test suite passing the standard ONNX Node test dataset directly to the WebNN EP.
- [ ] 242. Validate `Add` node outputs against WASM CPU.
- [ ] 243. Validate `Conv2d` node outputs against WASM CPU.
- [ ] 244. Validate `MatMul` node outputs against WASM CPU.
- [ ] 245. Run tests using the `webnn-polyfill` in headless Chrome/Puppeteer.
- [ ] 246. Run tests natively on macOS Chrome with `--enable-features=WebMachineLearningNeuralNetwork`.
- [ ] 247. Run tests natively on Windows Edge with NPU support enabled.
- [ ] 248. Calculate acceptable numerical drift tolerances (e.g., 1e-4) to account for NPU-specific precision drops.
- [ ] 249. Create tests for every single pad mode (`constant`, `edge`, `reflect`).
- [ ] 250. Create tests for specific broadcast combinations (e.g., `[1, 3, 224, 224] + [3, 1, 1]`).
- [ ] 251. Test multi-output nodes (e.g., `Split`, `TopK` fallback) correctness.
- [ ] 252. Ensure memory is pristine after 1000 successive iterations (leak testing).
- [ ] 253. Build a fuzzing harness generating random ONNX graphs and ensuring the WebNN EP doesn't crash the browser.
- [ ] 254. Test dynamic batch sizes (changing input shape from `[1, ...]` to `[4, ...]`) without re-compiling the graph.
- [ ] 255. Verify asynchronous execution does not block CSS animations on the main thread.

### Phase 19: Framework & Tooling Integration

- [ ] 256. Allow `Transformers.js` pipelines to explicitly target WebNN (`device: 'webnn'`).
- [ ] 257. Hook WebNN capability checking into the `AutoConfig` loader.
- [ ] 258. Ensure `onnx9000.genai` can offload LLM MatMul blocks natively to the NPU.
- [ ] 259. Integrate with `onnx9000.optimum` CLI to allow testing WebNN equivalence directly from the command line (`onnx9000 test webnn model.onnx`).
- [ ] 260. Publish a diagnostic web page showing "WebNN Readiness" for a user's current browser.
- [ ] 261. Integrate with React Native/Expo (when WebNN ships to mobile WebViews).
- [ ] 262. Support WebNN EP configuration flags (e.g., setting execution priority).
- [ ] 263. Emit standard `onnxruntime` EP log formats for compatibility with legacy debugging tools.
- [ ] 264. Support importing generic ONNX JSON (via ORT) and building the WebNN graph.
- [ ] 265. Document the complete list of supported ops and their spec version in a generated Markdown file.

### Phase 20: Advanced API Features & Future Specs

- [ ] 266. Prepare for W3C WebNN API v2 (dynamic shapes natively).
- [ ] 267. Map ONNX `Loop` natively if WebNN introduces control flow APIs.
- [ ] 268. Map ONNX `If` natively to WebNN.
- [ ] 269. Support specialized WebNN `lstm` and `gru` builder functions for RNN models.
- [ ] 270. Support WebNN `builder.resample2d` explicitly for ONNX `Resize` operations.
- [ ] 271. Support nearest-neighbor interpolation in WebNN `resample2d`.
- [ ] 272. Support linear interpolation in WebNN `resample2d`.
- [ ] 273. Support `builder.gatherNd` if added to the WebNN spec.
- [ ] 274. Handle WebNN `logicalAnd/Or/Not` applied to multi-dimensional masks.
- [ ] 275. Map ONNX `CumSum` to NPU native execution (often tricky, might require scan algorithms).
- [ ] 276. Provide hooks for WebNN `builder.gruCell` mapping.
- [ ] 277. Provide hooks for WebNN `builder.lstmCell` mapping.
- [ ] 278. Support explicit data layout overriding during WebNN graph build (ignoring ONNX constraints).
- [ ] 279. Build an automated transpiler: `onnx9000-to-wgsl` for ops rejected by the WebNN context, ensuring no fallback to slow JS math.
- [ ] 280. Handle `uint32` data types in WebNN (often required for Gather indices).
- [ ] 281. Integrate `onnx9000.image` pre-processing natively into the WebNN graph (fusing Normalize/Resize ops into the NPU).
- [ ] 282. Expose `builder.triangular` for specialized causal masking if present.
- [ ] 283. Support executing multiple isolated WebNN contexts concurrently for multi-model web apps.
- [ ] 284. Implement fallback logic for WebNN unsupported `dilations` values in specific layers.
- [ ] 285. Support `builder.dequantizeLinear` executing specifically on NPU vector engines.
- [ ] 286. Map ONNX `SpaceToDepth` and `DepthToSpace` to WebNN if supported natively.
- [ ] 287. Compile and run YOLO-v8 fully accelerated on the WebNN EP.
- [ ] 288. Compile and run MobileViT fully accelerated on the WebNN EP.
- [ ] 289. Compile and run Whisper (Encoder) fully accelerated on the WebNN EP.
- [ ] 290. Maintain an architecture compatibility matrix tracking exact NPU support levels (Qualcomm vs Apple vs Intel).
- [ ] 291. Validate exact compliance with WebNN Draft Spec W3C Working Drafts.
- [ ] 292. Support `builder.concat` with more than 5 inputs (handling NPU argument limits).
- [ ] 293. Track and bypass known WebNN Polyfill bugs dynamically.
- [ ] 294. Optimize constant memory uploads to prevent Chrome UI freezes during `builder.build()`.
- [ ] 295. Execute deep layout analysis (NCHW to NHWC) eliminating redundant transpose chains specific to NPU backends.
- [ ] 296. Map ONNX `HardSwish` natively using WebNN arithmetic `x * hardSigmoid`.
- [ ] 297. Support WebNN native `builder.softplus`.
- [ ] 298. Validate precise execution parity between `device: 'webgpu'` and `device: 'webnn'` on the exact same hardware.
- [ ] 299. Write comprehensive tutorial: "Deploying ONNX Models to NPUs in the Browser".
- [ ] 300. Release v1.0 complete feature parity certification matching the official C++ ONNX Runtime WebNN EP.
