# ONNX32: onnx2tf (Web-Native TFLite & EdgeTPU Exporter)

## Original Project Description

`onnx2tf` (often associated with PINTO0309's widely used repository) is a critical community tool for converting ONNX models into TensorFlow (`SavedModel`) and TensorFlow Lite (`.tflite`) formats. It heavily relies on a massive native Python TensorFlow installation and ONNX Runtime to parse graphs, calculate shapes, and meticulously translate layout structures (since ONNX uses `NCHW` channel-first layouts and TensorFlow/TFLite strongly prefers `NHWC` channel-last layouts). This tool is essential for taking standard AI models and deploying them onto mobile devices (Android NNAPI, iOS CoreML) and hardware accelerators like the Google Coral EdgeTPU.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

Instead of relying on Google's multi-gigabyte C++ TensorFlow framework to compile `.tflite` files, `onnx9000.onnx2tf` provides a **100% pure TypeScript and Python FlatBuffer compiler**.

- **Zero-Dependency Binary Emission:** It parses the ONNX graph and writes the TFLite FlatBuffer binary directly in memory, byte-by-byte. No `tensorflow`, `tflite`, or `flatc` compiler installations are required.
- **Browser-Based EdgeTPU Compilation:** Developers can drop an ONNX file into a web browser, and the library will natively transpose the graph to `NHWC` and generate a mobile-ready `.tflite` file instantly on the client side.
- **AOT Transposition:** Re-writing layouts (NCHW -> NHWC) is notoriously slow in Python. `onnx9000` uses its WASM-accelerated GraphSurgeon to permanently bake transpositions directly into the weights before export, guaranteeing peak performance on mobile DSPs without inference-time transposition overhead.
- **Unified Quantization:** Maps ONNX `QuantizeLinear`/`DequantizeLinear` directly to TFLite's asymmetric INT8 schema natively, preserving precision without requiring TF Lite's post-training quantization calibration tools.

---

## Exhaustive Implementation Checklist

### Phase 1: TFLite FlatBuffer Schema & Serialization Engine

- [ ] 1. Implement zero-dependency FlatBuffer Builder in TypeScript/JS.
- [ ] 2. Implement zero-dependency FlatBuffer Builder in Python.
- [ ] 3. Define TFLite `Model` root table schema natively.
- [ ] 4. Define TFLite `SubGraph` table schema natively.
- [ ] 5. Define TFLite `Tensor` table schema natively.
- [ ] 6. Define TFLite `Buffer` table schema natively.
- [ ] 7. Define TFLite `Operator` table schema natively.
- [ ] 8. Define TFLite `OperatorCode` table schema natively.
- [ ] 9. Define TFLite `QuantizationParameters` table schema.
- [ ] 10. Define TFLite `Metadata` table schema.
- [ ] 11. Implement TFLite version 3 header emission (`TFL3` magic bytes).
- [ ] 12. Implement strictly aligned memory writing (4-byte and 8-byte boundaries for buffers).
- [ ] 13. Support appending large binary weights directly to the `Buffer` array seamlessly.
- [ ] 14. Implement string serialization for Tensor and Operator names.
- [ ] 15. Handle Little-Endian binary encoding universally across all platforms (WASM/JS/Py).
- [ ] 16. Deduplicate identical operators in the `OperatorCode` array.
- [ ] 17. Deduplicate identical weight binaries in the `Buffer` array to save disk space.
- [ ] 18. Deduplicate empty/zero-byte buffers.
- [ ] 19. Ensure Buffer `0` is always strictly empty as required by the TFLite spec.
- [ ] 20. Track exact byte offsets during serialization to emit correct vtables.
- [ ] 21. Provide lazy buffer loading mapping from `onnx9000.Tensor` to FlatBuffer byte arrays.
- [ ] 22. Export structural JSON representation of the generated FlatBuffer for debugging.
- [ ] 23. Implement a TFLite FlatBuffer Reader (for bidirectional validation).
- [ ] 24. Validate generated `.tflite` files against standard `flatc` schema verifiers natively.
- [ ] 25. Support chunked writing for models exceeding JS `ArrayBuffer` limits (>2GB).
- [ ] 26. Extract ONNX `ModelProto` metadata (Producer, Version) to TFLite `Metadata` buffers.
- [ ] 27. Maintain deterministic output (identical ONNX = byte-for-byte identical TFLite).
- [ ] 28. Manage Javascript `BigInt` safely when writing 64-bit FlatBuffer offsets.
- [ ] 29. Emulate Python `struct.pack` efficiently in Javascript for primitive types.
- [ ] 30. Provide a validation pass ensuring no TFLite tensor exceeds standard device bounds.

### Phase 2: Global Layout Transposition (NCHW -> NHWC)

- [ ] 31. Implement AST Graph Pass: Identify all spatial convolutions and pooling ops.
- [ ] 32. Inject `Transpose` (`[0, 2, 3, 1]`) before every 4D spatial operation.
- [ ] 33. Inject `Transpose` (`[0, 3, 1, 2]`) after every 4D spatial operation.
- [ ] 34. Implement `Transpose` Push-Down: Move transpositions through elementwise ops (`Add`, `Mul`, `Relu`).
- [ ] 35. Implement `Transpose` Push-Down through `Concat` and `Split` (adjusting axes dynamically).
- [ ] 36. Implement `Transpose` Push-Down through `Reshape` (symbolically recalculating reshape targets).
- [ ] 37. Implement `Transpose` Cancellation: Eliminate adjacent `NCHW->NHWC` and `NHWC->NCHW` pairs.
- [ ] 38. Fold `Transpose` operations directly into `Constant` / `Initializer` weights statically in memory.
- [ ] 39. Support 1D layout conversion (`NCW` -> `NWC`).
- [ ] 40. Support 3D Video layout conversion (`NCDHW` -> `NDHWC`).
- [ ] 41. Handle ONNX `BatchNormalization` natively on NHWC layouts.
- [ ] 42. Map Keras/TF.js specific layout formats accurately if originating from `onnx9000.keras`.
- [ ] 43. Handle arbitrary `Expand` and `Tile` permutations during layout shift.
- [ ] 44. Generate explicit warnings if an irreducible Transpose node is left in the graph (hurts EdgeTPU).
- [ ] 45. Automatically recalculate all `ValueInfo` shapes topologically after layout mutation.
- [ ] 46. Support `--keep-nchw` flag for specific ops that TFLite supports natively in NCHW (though rare).
- [ ] 47. Translate ONNX `axis` parameters accurately for `Softmax` post-layout shift.
- [ ] 48. Translate ONNX `axis` parameters for `Gather` and `Scatter`.
- [ ] 49. Handle `ReduceMean` / `ReduceSum` spatial axes translations (`[2, 3]` -> `[1, 2]`).
- [ ] 50. Transpose Weight tensors explicitly for `Conv2D` (`[O, I, H, W]` -> `[O, H, W, I]`).
- [ ] 51. Transpose Weight tensors explicitly for `DepthwiseConv2D` (`[1, C, H, W]` -> `[1, H, W, C]`).
- [ ] 52. Transpose Weight tensors explicitly for `Conv2DTranspose`.
- [ ] 53. Ensure scalar biases are preserved correctly without layout corruption.
- [ ] 54. Verify dimension indexing stability for dynamic batch sizes (`-1`) during layout shifts.

### Phase 3: TFLite Tensor & Memory Mapping

- [ ] 55. Map ONNX `FLOAT` -> TFLite `FLOAT32`.
- [ ] 56. Map ONNX `FLOAT16` -> TFLite `FLOAT16`.
- [ ] 57. Map ONNX `INT32` -> TFLite `INT32`.
- [ ] 58. Map ONNX `INT64` -> TFLite `INT64`.
- [ ] 59. Map ONNX `INT8` -> TFLite `INT8`.
- [ ] 60. Map ONNX `UINT8` -> TFLite `UINT8`.
- [ ] 61. Map ONNX `BOOL` -> TFLite `BOOL`.
- [ ] 62. Map ONNX `STRING` -> TFLite `STRING`.
- [ ] 63. Handle ONNX `DOUBLE` (Float64) gracefully (downcast to Float32, as TFLite prefers Float32).
- [ ] 64. Map empty ONNX shapes `[]` to TFLite scalar shapes `[]`.
- [ ] 65. Map dynamic ONNX shapes `[-1, 224, 224, 3]` safely.
- [ ] 66. Emit `ShapeSignature` vectors for TFLite dynamic shapes.
- [ ] 67. Map ONNX Input Tensors to SubGraph `inputs` array.
- [ ] 68. Map ONNX Output Tensors to SubGraph `outputs` array.
- [ ] 69. Resolve ONNX Initializers directly to TFLite `Buffer` indices.
- [ ] 70. Generate unique integer IDs sequentially for all tensors.
- [ ] 71. Pack boolean ONNX tensors into TFLite bit-vectors if explicitly required.
- [ ] 72. Ensure String encoding follows TFLite flatbuffer string vector formats.
- [ ] 73. Provide fallback casting (`Cast`) automatically if TFLite lacks an op signature for a specific type.
- [ ] 74. Map 0-dimensional tensors (Scalars) consistently.

### Phase 4: Basic Arithmetic & Elementwise Mapping

- [ ] 75. Emit `ADD` (TFLite BuiltinOperator).
- [ ] 76. Emit `SUB`.
- [ ] 77. Emit `MUL`.
- [ ] 78. Emit `DIV`.
- [ ] 79. Emit `FLOOR_DIV`.
- [ ] 80. Emit `FLOOR_MOD` / `MOD`.
- [ ] 81. Emit `MAXIMUM`.
- [ ] 82. Emit `MINIMUM`.
- [ ] 83. Emit `POW`.
- [ ] 84. Emit `ABS`.
- [ ] 85. Emit `EXP`.
- [ ] 86. Emit `LOG`.
- [ ] 87. Emit `SQRT`.
- [ ] 88. Emit `RSQRT` (Reciprocal Square Root).
- [ ] 89. Emit `SIN`.
- [ ] 90. Emit `COS`.
- [ ] 91. Emit `NEG` (Negative).
- [ ] 92. Emit `CEIL`.
- [ ] 93. Emit `FLOOR`.
- [ ] 94. Emit `ROUND`.
- [ ] 95. Emit `SIGN`.
- [ ] 96. Handle ONNX implicit broadcasting natively matching TFLite broadcast rules.
- [ ] 97. Inject TFLite `BROADCAST_TO` explicitly if TFLite strict versions require explicit broadcasts.
- [ ] 98. Ensure TFLite `fused_activation_function` is utilized for `Add`+`Relu`, `Mul`+`Relu` optimizations.
- [ ] 99. Verify scalar vs tensor addition signatures map correctly to TFLite options.
- [ ] 100. Handle division by zero constraints if mathematically determinable during translation.

### Phase 5: Convolution & Spatial Mapping

- [ ] 101. Emit `CONV_2D`.
- [ ] 102. Extract ONNX `strides` to TFLite `stride_h`, `stride_w`.
- [ ] 103. Extract ONNX `dilations` to TFLite `dilation_h_factor`, `dilation_w_factor`.
- [ ] 104. Map ONNX explicit padding `[x1, y1, x2, y2]` to TFLite explicit padding if supported.
- [ ] 105. Detect and map symmetric padding to TFLite `PADDING_SAME`.
- [ ] 106. Detect and map zero padding to TFLite `PADDING_VALID`.
- [ ] 107. Inject `PAD` operations dynamically prior to `CONV_2D` if asymmetric padding cannot be expressed in TFLite natively.
- [ ] 108. Emit `DEPTHWISE_CONV_2D`.
- [ ] 109. Evaluate ONNX `group` attribute to trigger Depthwise translation natively.
- [ ] 110. Set `depth_multiplier` correctly for `DEPTHWISE_CONV_2D`.
- [ ] 111. Emit `TRANSPOSE_CONV` (Conv2DTranspose).
- [ ] 112. Map ONNX `output_padding` to TFLite exact output shape tensors.
- [ ] 113. Emit `MAX_POOL_2D`.
- [ ] 114. Extract pool `filter_height`, `filter_width`.
- [ ] 115. Emit `AVERAGE_POOL_2D`.
- [ ] 116. Map ONNX `GlobalAveragePool` to TFLite `MEAN` with spatial axes `[1, 2]`.
- [ ] 117. Map ONNX `GlobalMaxPool` to TFLite `REDUCE_MAX` with spatial axes `[1, 2]`.
- [ ] 118. Handle 1D Convolutions by expanding dimensions to 2D internally (`H=1`).
- [ ] 119. Handle 1D Pooling by expanding dimensions to 2D internally (`H=1`).
- [ ] 120. Emit `L2_POOL_2D`.
- [ ] 121. Handle Conv biases properly (must be 1D tensors matching output channels).
- [ ] 122. Support TFLite `fused_activation_function` in `CONV_2D` natively (ReLU, ReLU6, None).
- [ ] 123. Optimize `BatchNormalization` natively into Conv weights (folding) prior to TFLite export.
- [ ] 124. Throw warning for 3D Convolutions (`CONV_3D`) if targeting TFLite environments that lack 3D support.
- [ ] 125. Emit `CONV_3D` exclusively for TFLite Flex delegates or experimental spec configurations.

### Phase 6: Activations & Normalization Mapping

- [ ] 126. Emit `RELU`.
- [ ] 127. Emit `RELU6` (Map ONNX `Clip` with `0.0` to `6.0`).
- [ ] 128. Emit `LEAKY_RELU` (Parsing `alpha` parameter).
- [ ] 129. Emit `ELU`.
- [ ] 130. Emit `LOGISTIC` (Sigmoid).
- [ ] 131. Emit `TANH`.
- [ ] 132. Emit `SOFTMAX`.
- [ ] 133. Parse ONNX `axis` for `Softmax` and map to TFLite (defaulting to `-1`).
- [ ] 134. Emit `LOG_SOFTMAX`.
- [ ] 135. Emit `HARD_SWISH`.
- [ ] 136. Map ONNX `Gelu` to TFLite `GELU` (Builtin if available).
- [ ] 137. Map ONNX `Gelu` (Approximate) to `GELU` approximation math subgraph if builtin missing.
- [ ] 138. Emit `PRelu` (Parametric ReLU).
- [ ] 139. Map ONNX `BatchNormalization` to TFLite math operations (`Sub`, `Mul`, `Add`) if unfused.
- [ ] 140. Map ONNX `InstanceNormalization` to TFLite math subgraph or custom op.
- [ ] 141. Map ONNX `LayerNormalization` to TFLite builtin if available, otherwise subgraph.
- [ ] 142. Map ONNX `LpNormalization` to TFLite `L2_NORMALIZATION`.
- [ ] 143. Emit `LOCAL_RESPONSE_NORMALIZATION` (LRN).
- [ ] 144. Ensure fused activation bounds respect asymmetric INT8 limits natively.
- [ ] 145. Strip `Dropout` identity layers permanently from TFLite payload.

### Phase 7: Array & Shape Manipulation Mapping

- [ ] 146. Emit `RESHAPE`.
- [ ] 147. Provide exact `new_shape` options in TFLite builder.
- [ ] 148. Emit `TRANSPOSE`.
- [ ] 149. Emit `SQUEEZE` (Parsing `squeeze_dims`).
- [ ] 150. Emit `EXPAND_DIMS` (Map from ONNX `Unsqueeze`).
- [ ] 151. Emit `CONCATENATION`.
- [ ] 152. Parse `axis` for Concat and encode into options.
- [ ] 153. Emit `SPLIT`.
- [ ] 154. Emit `SPLIT_V` (for uneven splits).
- [ ] 155. Emit `SLICE`.
- [ ] 156. Emit `STRIDED_SLICE` (Mapping complex ONNX Slices with strides/steps).
- [ ] 157. Encode `begin_mask`, `end_mask`, `shrink_axis_mask` natively for `STRIDED_SLICE`.
- [ ] 158. Emit `GATHER`.
- [ ] 159. Emit `GATHER_ND`.
- [ ] 160. Emit `SCATTER_ND`.
- [ ] 161. Map ONNX `ScatterElements` to specific TFLite equivalents or mathematical subgraphs.
- [ ] 162. Emit `TILE`.
- [ ] 163. Emit `PAD`.
- [ ] 164. Emit `PADV2` (Handling constant values).
- [ ] 165. Emit `MIRROR_PAD` (Handling Reflect and Edge padding).
- [ ] 166. Emit `SHAPE` (Map ONNX Shape).
- [ ] 167. Emit `PACK` (Map ONNX sequence logic or Stack).
- [ ] 168. Emit `UNPACK` (Map ONNX Unstack/Split).
- [ ] 169. Map ONNX `ConstantOfShape` to TFLite `FILL`.
- [ ] 170. Map ONNX `Expand` to TFLite `BROADCAST_TO`.

### Phase 8: Matrix Multiplication & Linear Algebra

- [ ] 171. Emit `FULLY_CONNECTED`.
- [ ] 172. Evaluate ONNX `Gemm` dimensions to determine if it maps to `FULLY_CONNECTED`.
- [ ] 173. Evaluate ONNX `MatMul` + `Add` patterns to fuse into `FULLY_CONNECTED`.
- [ ] 174. Set `keep_num_dims` options dynamically in TFLite options.
- [ ] 175. Handle weight transpositions required by TFLite `FULLY_CONNECTED` (`[I, O]` vs `[O, I]`).
- [ ] 176. Emit `BATCH_MATMUL`.
- [ ] 177. Configure `adj_x` and `adj_y` natively based on ONNX transpose structures.
- [ ] 178. Handle implicit `Einsum` equations via `Reshape` and `BATCH_MATMUL` decomposition.
- [ ] 179. Emit `MATRIX_DIAG`.
- [ ] 180. Emit `MATRIX_SET_DIAG`.

### Phase 9: Logical, Reduction, & Control Flow Mapping

- [ ] 181. Emit `EQUAL`.
- [ ] 182. Emit `NOT_EQUAL`.
- [ ] 183. Emit `LESS`.
- [ ] 184. Emit `LESS_EQUAL`.
- [ ] 185. Emit `GREATER`.
- [ ] 186. Emit `GREATER_EQUAL`.
- [ ] 187. Emit `LOGICAL_AND`.
- [ ] 188. Emit `LOGICAL_OR`.
- [ ] 189. Emit `LOGICAL_NOT`.
- [ ] 190. Emit `WHERE` (Select / SelectV2).
- [ ] 191. Emit `REDUCE_MEAN`.
- [ ] 192. Emit `REDUCE_MAX`.
- [ ] 193. Emit `REDUCE_MIN`.
- [ ] 194. Emit `REDUCE_PROD`.
- [ ] 195. Emit `SUM` (ReduceSum).
- [ ] 196. Emit `REDUCE_ANY` (Logical Or reduction).
- [ ] 197. Emit `REDUCE_ALL` (Logical And reduction).
- [ ] 198. Map ONNX `If` to TFLite `IF` control flow operators.
- [ ] 199. Extract SubGraphs iteratively into the TFLite Flatbuffer to support `IF` branches.
- [ ] 200. Map ONNX `Loop` to TFLite `WHILE` loops.

### Phase 10: Advanced Vision & Sorting Ops

- [ ] 201. Emit `RESIZE_BILINEAR`.
- [ ] 202. Encode `align_corners` and `half_pixel_centers` correctly.
- [ ] 203. Emit `RESIZE_NEAREST_NEIGHBOR`.
- [ ] 204. Map ONNX `Resize` scaling arrays explicitly into TFLite static shape tensors.
- [ ] 205. Emit `SPACE_TO_DEPTH`.
- [ ] 206. Encode `block_size` attribute securely.
- [ ] 207. Emit `DEPTH_TO_SPACE`.
- [ ] 208. Emit `SPACE_TO_BATCH_ND`.
- [ ] 209. Emit `BATCH_TO_SPACE_ND`.
- [ ] 210. Emit `ARG_MAX`.
- [ ] 211. Emit `ARG_MIN`.
- [ ] 212. Emit `TOPK_V2`.
- [ ] 213. Emit `UNIQUE`.
- [ ] 214. Emit `REVERSE_V2`.
- [ ] 215. Map ONNX `CumSum` to TFLite `CUMSUM`.
- [ ] 216. Map ONNX `NonMaxSuppression` to TFLite `NON_MAX_SUPPRESSION_V4` / `V5`.
- [ ] 217. Emit `CUMPROD` natively if TFLite schema supports it.
- [ ] 218. Map ONNX `GridSample` to TFLite custom or math equivalents.
- [ ] 219. Emit `SEGMENT_SUM`.
- [ ] 220. Support TFLite specialized `LSH_PROJECTION`.

### Phase 11: RNN, LSTM, & Sequence Mapping

- [ ] 221. Emit `RNN`.
- [ ] 222. Emit `UNIDIRECTIONAL_SEQUENCE_RNN`.
- [ ] 223. Emit `LSTM`.
- [ ] 224. Emit `UNIDIRECTIONAL_SEQUENCE_LSTM`.
- [ ] 225. Parse ONNX LSTM input gates, peepholes, and weights into TFLite's massive flattened tensor requirements.
- [ ] 226. Support `time_major` flags natively.
- [ ] 227. Emit `BIDIRECTIONAL_SEQUENCE_LSTM`.
- [ ] 228. Split ONNX bidirectional weights into Forward and Backward explicitly for TFLite.
- [ ] 229. Emit `GRU` / `UNIDIRECTIONAL_SEQUENCE_GRU`.
- [ ] 230. Support Stateful TFLite Execution (Variable tensors) if sequence history requires persistence.

### Phase 12: Quantization (TFLite Int8 / UINT8 / FP16)

- [ ] 231. Encode `QuantizationParameters` table natively.
- [ ] 232. Support `scale` (Float array) definitions.
- [ ] 233. Support `zero_point` (Int64 array) definitions.
- [ ] 234. Map ONNX `QuantizeLinear` directly to TFLite `QUANTIZE`.
- [ ] 235. Map ONNX `DequantizeLinear` directly to TFLite `DEQUANTIZE`.
- [ ] 236. Generate explicit Asymmetric INT8 TFLite models natively from ONNX QDQ topologies.
- [ ] 237. Produce explicit Per-Channel quantization arrays (1D scales/zeros for DepthwiseConvs).
- [ ] 238. Extract `quantized_dimension` correctly for Per-Channel ops.
- [ ] 239. Handle legacy TFLite `UINT8` quantization generation.
- [ ] 240. Ensure INT16x8 (16-bit activations, 8-bit weights) metadata can be encoded natively.
- [ ] 241. Downcast `FLOAT32` FlatBuffer arrays entirely to `FLOAT16` bytes explicitly for FP16 models.
- [ ] 242. Set `FLOAT16` tensor type explicitly in the `Tensor` schema.
- [ ] 243. Identify standard fake-quantize sequences in ONNX and convert directly to Int8 TFLite tensors natively.
- [ ] 244. Implement MinMax parsing to embed fallback quantization metadata inside TFLite.
- [ ] 245. Validate resulting quantized schema against EdgeTPU compiler requirements natively.

### Phase 13: TensorFlow SavedModel (Protobuf) Generator

- [ ] 246. Implement zero-dependency `saved_model.pb` Protobuf generator in TS/Python.
- [ ] 247. Define TF `GraphDef` schema natively.
- [ ] 248. Define TF `SignatureDef` schema natively.
- [ ] 249. Define TF `SavedModel` structural properties.
- [ ] 250. Map ONNX graph into TF `NodeDef` lists natively.
- [ ] 251. Map ONNX Initializers directly to TF `Const` nodes.
- [ ] 252. Generate standard TF `variables.data-00000-of-00001` binary payloads explicitly.
- [ ] 253. Generate standard TF `variables.index` (SSTable format) natively in JS/Python.
- [ ] 254. Write `saved_model/` directory structure entirely in a JSZip blob for easy browser download.
- [ ] 255. Support `serving_default` tag bindings for strict TF Serving compatibility.
- [ ] 256. Handle TF1/TF2 legacy bridging markers inside the SavedModel.
- [ ] 257. Extract ONNX strings to TF `DT_STRING` records.
- [ ] 258. Convert ONNX dynamic shapes to `Dim` nodes with `size: -1` in the TF Protobuf.
- [ ] 259. Map custom domains securely into TF `CustomOp` definitions.
- [ ] 260. Output the raw `saved_model` bundle instantly to the local filesystem via CLI.

### Phase 14: EdgeTPU & NNAPI Specific Optimizations

- [ ] 261. Inject padding specifically to satisfy EdgeTPU dimension multiples (e.g., channels multiple of 8 or 4).
- [ ] 262. Verify strict Full-Integer INT8 quantization compliance (no Float32 nodes left anywhere) to prevent EdgeTPU fallback to CPU.
- [ ] 263. Analyze TFLite execution plan natively to identify operations that will break NNAPI compatibility.
- [ ] 264. Avoid generating `StridedSlice` with dynamic offsets (EdgeTPU hates this).
- [ ] 265. Rewrite `Softmax` on EdgeTPU using standard Taylor expansion math graphs if native is unsupported.
- [ ] 266. Emulate `LeakyRelu` on older NNAPI targets using `Maximum(x, alpha * x)`.
- [ ] 267. Expand `MatMul` into `FullyConnected` + `Reshape` consistently for edge devices.
- [ ] 268. Replace 1D Convolutions dynamically with 2D Convolutions for mobile DSP compatibility.
- [ ] 269. Eliminate complex Broadcasts on edge targets by expanding tensors statically before serialization.
- [ ] 270. Issue detailed "EdgeTPU Compatibility Report" upon TFLite export completion.

### Phase 15: TFLite Custom Ops & Builtin Signatures

- [ ] 271. Implement TFLite Custom Operator embedding in FlatBuffers (handling arbitrary string names).
- [ ] 272. Map ONNX `NonMaxSuppression` to standard TFLite `TFLite_Detection_PostProcess` custom op.
- [ ] 273. Support Flex Delegates (`Select TF` ops) embedding TF operators within TFLite flatbuffers natively.
- [ ] 274. Handle versioning of TFLite Builtin Operators (e.g., `ADD` version 1 vs version 2 for broadcast support).
- [ ] 275. Automatically bump TFLite op versions based on ONNX feature usage dynamically.
- [ ] 276. Encode `custom_options` byte arrays securely for proprietary hardware runtimes.
- [ ] 277. Strip experimental custom ops optionally to produce a "clean" TFLite file.
- [ ] 278. Inject MediaPipe specific metadata blocks into TFLite optionally.
- [ ] 279. Support TFLite Micro target generation (stripping unnecessary headers for tiny microcontrollers).
- [ ] 280. Add support for creating multi-signature TFLite models.

### Phase 16: CLI & Build Tooling (`onnx9000 onnx2tf`)

- [ ] 281. Implement CLI: `onnx9000 onnx2tf model.onnx -o model.tflite`.
- [ ] 282. Add `--int8` flag triggering quantization natively during export.
- [ ] 283. Add `--fp16` flag.
- [ ] 284. Add `--saved-model` flag to output full TF directories.
- [ ] 285. Add `--dynamic-batch` handling explicitly via CLI overrides (`-b 1`).
- [ ] 286. Add `--keep-nchw` override flag.
- [ ] 287. Implement progress bars for compiling massive flatbuffers sequentially.
- [ ] 288. Support processing ONNX models with external `.bin` weights natively.
- [ ] 289. Provide `--disable-optimization` flag.
- [ ] 290. Establish unit test parity checking TFLite CLI parameters matching PINTO0309's standard scripts.

### Phase 17: Web UI (The Universal Browser Converter)

- [ ] 291. Build a static Vue/React page "ONNX to TFLite Converter".
- [ ] 292. Provide drag-and-drop ingestion of `model.onnx`.
- [ ] 293. Provide toggle switches for "Quantize Int8", "FP16", "Optimize for EdgeTPU".
- [ ] 294. Utilize Web Workers to perform the AST traversal and FlatBuffer building without blocking the main UI.
- [ ] 295. Stream the generated `.tflite` directly to a local Blob Download.
- [ ] 296. Offer an embedded interactive graph visualizer (Netron style) showing the final TFLite layout.
- [ ] 297. Show exact memory payload reduction when enabling quantization features via UI.
- [ ] 298. Display detailed error messages directly in the DOM if ONNX topologies contain unsupported operators.
- [ ] 299. Ensure WebAssembly memory bounds are handled securely to prevent DOM crashes on 2GB+ files.
- [ ] 300. Maintain absolute zero-server contact (100% privacy preserving client-side compilation).

### Phase 18: End-to-End Testing & Regression Validations

- [ ] 301. Unit Test: Convert ONNX ResNet50 -> TFLite -> Run via WASM TF Lite Interpreter.
- [ ] 302. Unit Test: Convert ONNX MobileNetV2 -> TFLite -> Validate exact Cosine Similarity.
- [ ] 303. Unit Test: Convert ONNX YOLOv8 -> TFLite -> Validate bounding boxes.
- [ ] 304. Unit Test: Convert ONNX Whisper -> TFLite -> Validate audio transcriptions.
- [ ] 305. Unit Test: Validate multi-output branch shapes in DeepLabV3.
- [ ] 306. Check numerical accuracy of NCHW to NHWC layout modifications natively.
- [ ] 307. Fuzz test the FlatBuffer writer against intentionally corrupted ONNX proto files.
- [ ] 308. Verify memory leak absence when processing 100+ files sequentially in Node.js.
- [ ] 309. Ensure exact byte equivalence with Google's native `TFLiteConverter` output for identical graph structures.
- [ ] 310. Measure compilation time (Target: < 5 seconds for a 500MB ONNX model on a standard M1 Mac via Node.js).

### Phase 19: Edge Cases & Quirks

- [ ] 311. Handle implicit ONNX Shape broadcasting against empty tensors successfully.
- [ ] 312. Rewrite negative axis references statically to positive axis offsets during conversion to prevent TFLite runtime crashes.
- [ ] 313. Resolve TensorFlow's strict shape requirements for `Concat` (must have same ranks).
- [ ] 314. Prevent `Int64` tensor generation inside mobile targets (converting natively to `Int32` and warning user).
- [ ] 315. Manage explicitly unknown spatial sizes (`[1, -1, -1, 3]`) natively.
- [ ] 316. Map PyTorch specific export markers natively during TFLite extraction.
- [ ] 317. Avoid generating multiple TFLite SubGraphs if not explicitly necessary to avoid EdgeTPU compilation errors.
- [ ] 318. Emulate ONNX `Einsum` explicitly into transposes and batch-matmuls natively prior to TFLite injection.
- [ ] 319. Catch nested loops (`Loop` inside `If`) and warn users about severe mobile performance degradation.
- [ ] 320. Provide fallback mappings for HuggingFace Tokenizer custom nodes inside the generic ONNX graph.

### Phase 20: Delivery & Documentation

- [ ] 321. Provide comprehensive documentation: "Deploying ONNX models to Android using `onnx9000`".
- [ ] 322. Provide documentation: "Compiling ONNX for Coral EdgeTPU via the Browser".
- [ ] 323. Establish specific GitHub Issue templates for `onnx2tf` conversion failures.
- [ ] 324. Release as an independent NPM module `@onnx9000/tflite-exporter`.
- [ ] 325. Setup automated GitHub actions testing integration against EdgeTPU Compiler binaries.
- [ ] 326. Ensure TypeScript definition files (`.d.ts`) accurately reflect the FlatBuffer configurations.
- [ ] 327. Provide explicit `Buffer` cleanup operations to satisfy rigorous JS memory lifecycles.
- [ ] 328. Output detailed debugging metadata optionally alongside the `.tflite` binary.
- [ ] 329. Allow custom `tflite` quantization schema extensions manually via JS API arguments.
- [ ] 330. Guarantee final v1.0 feature parity with the original Python `onnx2tf` project natively in TS/WASM.
