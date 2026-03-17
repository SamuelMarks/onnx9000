# ONNX28: keras2onnx & tfjs-to-onnx (Web-Native Keras Converter)

## Original Project Description

`keras2onnx` (and its underlying dependencies like `tf2onnx`) is a Python-based conversion tool that translates Keras and TensorFlow models into the standard ONNX format. It parses Keras `.h5`, `.keras`, or SavedModel files, extracting the computational graph and weight tensors, and meticulously maps Keras layer semantics (which default to NHWC layout) into ONNX operator semantics (which default to NCHW layout). It requires a heavy, full-scale Python installation with TensorFlow and ONNX pip packages installed.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

`onnx9000.keras` eliminates the Python dependency completely, offering a **pure TypeScript/WebAssembly converter**.

- **Browser-Based Conversion:** Developers can drag and drop a Keras `.h5` file or a TensorFlow.js `model.json` directly into the browser, and `onnx9000` will output a `.onnx` file instantly, strictly client-side.
- **Dual-Format Ingestion:** Natively parses both Keras HDF5 (`.h5`) via a pure-JS HDF5 reader, and TensorFlow.js formats (`LayersModel` and `GraphModel`), providing an automatic bridge from the JS ecosystem to ONNX.
- **Zero-Copy Weight Transposition:** Translating Keras weight layouts (e.g., `[H, W, In, Out]`) to ONNX layouts (`[Out, In, H, W]`) is performed by highly optimized WASM kernels to prevent browser tab crashes on massive models.
- **Direct Execution Pipeline:** Converted models can be instantly routed into the `onnx9000` execution backend (WebGPU/WASM), allowing a user to `await onnx9000.keras.load('model.json')` and run it as if it were a native ONNX model.

---

## Exhaustive Implementation Checklist

### Phase 1: Ingestion & Format Parsing (TF.js & Keras H5)

- [ ] 1. Implement `tfjs.LayersModel` (`model.json`) JSON schema parser.
- [ ] 2. Implement `tfjs.GraphModel` JSON schema parser.
- [ ] 3. Implement external binary weight shard downloader/parser for TF.js.
- [ ] 4. Combine chunked `.bin` shards into a contiguous ArrayBuffer.
- [ ] 5. Map TF.js weight manifests to specific layer variables.
- [ ] 6. Implement pure-JS HDF5 (`.h5`) file reader.
- [ ] 7. Parse Keras `model_config` JSON strings embedded within HDF5 files.
- [ ] 8. Extract layer weights sequentially from HDF5 datasets.
- [ ] 9. Implement parser for the newer Keras 3 `.keras` zip-based format.
- [ ] 10. Support reading from local `File`/`Blob` objects in the browser.
- [ ] 11. Support fetching from remote URLs (with CORS handling).
- [ ] 12. Extract input specifications (shapes, names, types) from Keras config.
- [ ] 13. Extract output specifications from Keras config.
- [ ] 14. Identify multi-input / multi-output model topologies.
- [ ] 15. Build an internal abstract graph of Keras layers before ONNX translation.

### Phase 2: Core Layout Translation Engine (NHWC to NCHW)

- [ ] 16. Build the NHWC (Channels Last) to NCHW (Channels First) shape translator.
- [ ] 17. Build the `onnx9000` Transpose WASM kernel for 4D Image weights (Conv2D: `[H, W, I, O]` -> `[O, I, H, W]`).
- [ ] 18. Build the Transpose WASM kernel for 3D Sequence weights.
- [ ] 19. Build the Transpose WASM kernel for 5D Video weights (Conv3D).
- [ ] 20. Transpose Keras `Dense` weights (`[In, Out]` -> `[Out, In]`) where ONNX requires explicit GEMM mapping.
- [ ] 21. Track layout states dynamically (inserting ONNX `Transpose` ops dynamically if a Keras layer explicitly assumes NHWC data).
- [ ] 22. Implement a layout optimizer pass to remove redundant `Transpose` (e.g., `NCHW->NHWC->NCHW` collapses).
- [ ] 23. Handle Keras `data_format="channels_first"` layers gracefully (bypassing transpose injection).
- [ ] 24. Resolve Spatial padding discrepancies between Keras and ONNX.
- [ ] 25. Convert explicit TF `padding="SAME"` behavior to explicit ONNX padding values.
- [ ] 26. Convert explicit TF `padding="VALID"` behavior to ONNX padding values.

### Phase 3: Core Keras Layers Mapping (ONNX Emitters)

- [ ] 27. Map `InputLayer` to ONNX Graph Inputs.
- [ ] 28. Map `Dense` to ONNX `MatMul` + `Add` (Bias) or `Gemm`.
- [ ] 29. Extract `Dense` activation and append matching ONNX activation node.
- [ ] 30. Map `Activation` layer directly.
- [ ] 31. Map `ReLU` activation.
- [ ] 32. Map `Softmax` activation (handling axis conversions).
- [ ] 33. Map `LeakyReLU` activation.
- [ ] 34. Map `PReLU` activation (handling shared axes constraints).
- [ ] 35. Map `ELU` activation.
- [ ] 36. Map `ThresholdedReLU` activation.
- [ ] 37. Map `Softplus` activation.
- [ ] 38. Map `Softsign` activation.
- [ ] 39. Map `HardSigmoid` activation.
- [ ] 40. Map `Swish` / `SiLU` activation.
- [ ] 41. Map `GELU` activation (handling approx vs exact flags).
- [ ] 42. Map `Dropout` to ONNX `Identity` (or drop entirely for inference).
- [ ] 43. Map `SpatialDropout1D`, `SpatialDropout2D`, `SpatialDropout3D` to `Identity`.
- [ ] 44. Map `GaussianDropout` to `Identity`.
- [ ] 45. Map `GaussianNoise` to `Identity`.
- [ ] 46. Map `ActivityRegularization` to `Identity`.
- [ ] 47. Map `AlphaDropout` to `Identity`.

### Phase 4: Convolutional Layers Mapping

- [ ] 48. Map `Conv1D` to ONNX `Conv`.
- [ ] 49. Map `Conv2D` to ONNX `Conv`.
- [ ] 50. Map `Conv3D` to ONNX `Conv`.
- [ ] 51. Parse and apply `strides` tuple to ONNX.
- [ ] 52. Parse and apply `dilation_rate` tuple to ONNX.
- [ ] 53. Parse and apply `groups` attribute.
- [ ] 54. Map `SeparableConv1D` to Depthwise `Conv` + Pointwise `Conv`.
- [ ] 55. Map `SeparableConv2D` to Depthwise `Conv` + Pointwise `Conv`.
- [ ] 56. Map `DepthwiseConv2D` to ONNX `Conv` with `groups = in_channels`.
- [ ] 57. Map `Conv1DTranspose` to ONNX `ConvTranspose`.
- [ ] 58. Map `Conv2DTranspose` to ONNX `ConvTranspose`.
- [ ] 59. Map `Conv3DTranspose` to ONNX `ConvTranspose`.
- [ ] 60. Calculate ONNX `output_padding` dynamically to match Keras shape inference for Transpose Convs.

### Phase 5: Pooling Layers Mapping

- [ ] 61. Map `MaxPooling1D` to ONNX `MaxPool`.
- [ ] 62. Map `MaxPooling2D` to ONNX `MaxPool`.
- [ ] 63. Map `MaxPooling3D` to ONNX `MaxPool`.
- [ ] 64. Map `AveragePooling1D` to ONNX `AveragePool`.
- [ ] 65. Map `AveragePooling2D` to ONNX `AveragePool`.
- [ ] 66. Map `AveragePooling3D` to ONNX `AveragePool`.
- [ ] 67. Map `GlobalMaxPooling1D` to ONNX `GlobalMaxPool`.
- [ ] 68. Map `GlobalMaxPooling2D` to ONNX `GlobalMaxPool`.
- [ ] 69. Map `GlobalMaxPooling3D` to ONNX `GlobalMaxPool`.
- [ ] 70. Map `GlobalAveragePooling1D` to ONNX `GlobalAveragePool`.
- [ ] 71. Map `GlobalAveragePooling2D` to ONNX `GlobalAveragePool`.
- [ ] 72. Map `GlobalAveragePooling3D` to ONNX `GlobalAveragePool`.
- [ ] 73. Handle Keras `keepdims=False` (default in GlobalPools) by inserting ONNX `Squeeze`.

### Phase 6: Recurrent Layers (RNN/LSTM/GRU) Mapping

- [ ] 74. Map `SimpleRNN` to ONNX `RNN`.
- [ ] 75. Transpose and pack Keras RNN weights (`kernel`, `recurrent_kernel`, `bias`) into ONNX RNN combined weights `W` and `R`.
- [ ] 76. Handle Keras `return_sequences=True` (outputting full sequence).
- [ ] 77. Handle Keras `return_sequences=False` (outputting last state, slicing ONNX output).
- [ ] 78. Handle Keras `return_state=True` (outputting hidden states).
- [ ] 79. Map `LSTM` to ONNX `LSTM`.
- [ ] 80. Convert Keras LSTM weight gate order (i, f, c, o) to ONNX LSTM gate order (i, o, f, c).
- [ ] 81. Map `GRU` to ONNX `GRU`.
- [ ] 82. Convert Keras GRU weight gate order (z, r, h) to ONNX GRU gate order (z, r, h).
- [ ] 83. Handle GRU `reset_after` flag (mapping to linear_before_reset in ONNX).
- [ ] 84. Map `Bidirectional` wrapper for RNN/LSTM/GRU.
- [ ] 85. Combine forward and backward Keras weights into ONNX multi-directional weights.
- [ ] 86. Implement `merge_mode='concat'` for Bidirectional outputs.
- [ ] 87. Implement `merge_mode='sum'` for Bidirectional outputs.
- [ ] 88. Implement `merge_mode='mul'` for Bidirectional outputs.
- [ ] 89. Implement `merge_mode='ave'` for Bidirectional outputs.
- [ ] 90. Handle initial state inputs securely for stateful sequence models.

### Phase 7: Merge Layers Mapping

- [ ] 91. Map `Add` to ONNX `Add` (with multi-input accumulation).
- [ ] 92. Map `Subtract` to ONNX `Sub`.
- [ ] 93. Map `Multiply` to ONNX `Mul` (with multi-input accumulation).
- [ ] 94. Map `Average` to ONNX `Mean`.
- [ ] 95. Map `Maximum` to ONNX `Max`.
- [ ] 96. Map `Minimum` to ONNX `Min`.
- [ ] 97. Map `Concatenate` to ONNX `Concat`.
- [ ] 98. Resolve negative `axis` properly for `Concatenate` within the NHWC -> NCHW translation context.
- [ ] 99. Map `Dot` to ONNX `MatMul` (handling explicit axes parameters via Transpose injections).
- [ ] 100. Handle implicit broadcasting differences between Keras and ONNX during merge operations.

### Phase 8: Advanced & Attention Layers

- [ ] 101. Map `Attention` to explicit ONNX Subgraph (MatMul + Softmax + MatMul).
- [ ] 102. Handle causal masks dynamically inside `Attention` mapping.
- [ ] 103. Map `AdditiveAttention` (Bahdanau) to ONNX explicit Ops.
- [ ] 104. Map Keras 3 `MultiHeadAttention` to ONNX explicitly (or map to specific ONNX `Attention` op if supported by opset).
- [ ] 105. Split multi-head weights out of the Keras dense representations.
- [ ] 106. Handle `use_causal_mask` for MHA.
- [ ] 107. Map Keras `Embedding` layer to ONNX `Gather`.
- [ ] 108. Support `mask_zero=True` in `Embedding` by emitting an explicit boolean mask output.
- [ ] 109. Map `ConvLSTM1D` to explicit sequence of Conv + LSTM logic.
- [ ] 110. Map `ConvLSTM2D` to explicit sequence.
- [ ] 111. Map `ConvLSTM3D` to explicit sequence.
- [ ] 112. Map Keras `TimeDistributed` wrapper by reshaping `[batch, time, ...]` -> `[batch * time, ...]` -> Apply Layer -> Reshape back.

### Phase 9: Normalization & Reshaping Layers

- [ ] 113. Map `BatchNormalization` to ONNX `BatchNormalization`.
- [ ] 114. Extract moving mean, moving variance, beta, and gamma.
- [ ] 115. Map `LayerNormalization` to ONNX `LayerNormalization` or `ReduceMean`->`Sub`->`Pow`->`Add`->`Div` if ONNX opset is too low.
- [ ] 116. Handle `axis` mapping for `LayerNormalization`.
- [ ] 117. Map `UnitNormalization` to ONNX `LpNormalization`.
- [ ] 118. Map `GroupNormalization` to ONNX standard operations.
- [ ] 119. Map `Reshape` to ONNX `Reshape`.
- [ ] 120. Translate Keras implicit `-1` batch dimension inside `Reshape`.
- [ ] 121. Map `Flatten` to ONNX `Flatten`.
- [ ] 122. Handle `data_format` correctly inside `Flatten`.
- [ ] 123. Map `RepeatVector` to ONNX `Expand` or `Tile`.
- [ ] 124. Map `Permute` to ONNX `Transpose`.
- [ ] 125. Map `ZeroPadding1D` to ONNX `Pad`.
- [ ] 126. Map `ZeroPadding2D` to ONNX `Pad`.
- [ ] 127. Map `ZeroPadding3D` to ONNX `Pad`.
- [ ] 128. Map `Cropping1D` to ONNX `Slice`.
- [ ] 129. Map `Cropping2D` to ONNX `Slice`.
- [ ] 130. Map `Cropping3D` to ONNX `Slice`.
- [ ] 131. Map `UpSampling1D` to ONNX `Resize` (Nearest/Bilinear).
- [ ] 132. Map `UpSampling2D` to ONNX `Resize`.
- [ ] 133. Map `UpSampling3D` to ONNX `Resize`.

### Phase 10: TF.js GraphModel Specific Ops (tf.\* equivalents)

- [ ] 134. Map TF.js `tf.add` to ONNX `Add`.
- [ ] 135. Map TF.js `tf.sub` to ONNX `Sub`.
- [ ] 136. Map TF.js `tf.mul` to ONNX `Mul`.
- [ ] 137. Map TF.js `tf.div` to ONNX `Div`.
- [ ] 138. Map TF.js `tf.matMul` to ONNX `MatMul`.
- [ ] 139. Map TF.js `tf.square` to ONNX `Pow` (exponent 2).
- [ ] 140. Map TF.js `tf.sqrt` to ONNX `Sqrt`.
- [ ] 141. Map TF.js `tf.exp` to ONNX `Exp`.
- [ ] 142. Map TF.js `tf.log` to ONNX `Log`.
- [ ] 143. Map TF.js `tf.maximum` to ONNX `Max`.
- [ ] 144. Map TF.js `tf.minimum` to ONNX `Min`.
- [ ] 145. Map TF.js `tf.sum` to ONNX `ReduceSum`.
- [ ] 146. Map TF.js `tf.mean` to ONNX `ReduceMean`.
- [ ] 147. Map TF.js `tf.max` to ONNX `ReduceMax`.
- [ ] 148. Map TF.js `tf.min` to ONNX `ReduceMin`.
- [ ] 149. Map TF.js `tf.argMax` to ONNX `ArgMax`.
- [ ] 150. Map TF.js `tf.argMin` to ONNX `ArgMin`.
- [ ] 151. Map TF.js `tf.split` to ONNX `Split`.
- [ ] 152. Map TF.js `tf.concat` to ONNX `Concat`.
- [ ] 153. Map TF.js `tf.slice` to ONNX `Slice`.
- [ ] 154. Map TF.js `tf.stridedSlice` to ONNX `Slice` (translating end masks).
- [ ] 155. Map TF.js `tf.gather` to ONNX `Gather`.
- [ ] 156. Map TF.js `tf.gatherNd` to ONNX `GatherND`.
- [ ] 157. Map TF.js `tf.where` to ONNX `Where`.
- [ ] 158. Map TF.js `tf.tensorScatterUpdate` to ONNX `ScatterND`.
- [ ] 159. Map TF.js `tf.image.resizeBilinear` to ONNX `Resize`.
- [ ] 160. Map TF.js `tf.image.resizeNearestNeighbor` to ONNX `Resize`.

### Phase 11: End-to-End Validation (Vision Architectures)

- [ ] 161. Convert and validate TF.js `MobileNetV1`.
- [ ] 162. Convert and validate TF.js `MobileNetV2`.
- [ ] 163. Convert and validate TF.js `MobileNetV3`.
- [ ] 164. Convert and validate Keras `ResNet50`.
- [ ] 165. Convert and validate Keras `ResNet101`.
- [ ] 166. Convert and validate Keras `InceptionV3`.
- [ ] 167. Convert and validate Keras `Xception`.
- [ ] 168. Convert and validate Keras `VGG16`.
- [ ] 169. Convert and validate Keras `VGG19`.
- [ ] 170. Convert and validate Keras `EfficientNetB0` through `B7`.
- [ ] 171. Convert and validate Keras `DenseNet121`.
- [ ] 172. Convert and validate Keras `NASNetMobile`.
- [ ] 173. Convert and validate TF.js `PoseNet`.
- [ ] 174. Convert and validate TF.js `BodyPix`.
- [ ] 175. Verify 100% equivalent spatial output matrices against native TF.js execution (tolerance 1e-5).

### Phase 12: End-to-End Validation (NLP & Sequence Architectures)

- [ ] 176. Convert and validate TF.js `Universal Sentence Encoder` (USE).
- [ ] 177. Convert and validate Keras `Transformer` implementation (MultiHeadAttention).
- [ ] 178. Convert and validate TF.js `Toxicity` text classifier.
- [ ] 179. Convert and validate Keras `LSTM` character-level generator.
- [ ] 180. Convert and validate Keras `GRU` sequence-to-sequence model.
- [ ] 181. Verify dynamic sequence lengths compile cleanly to ONNX dynamic axes.
- [ ] 182. Handle Keras Embedding weights loading properly into ONNX Gather initializers.
- [ ] 183. Check precise parity of Bidirectional states outputs against TF.js.

### Phase 13: End-to-End Validation (Generative & Audio)

- [ ] 184. Convert and validate Keras `DCGAN` generator.
- [ ] 185. Convert and validate Keras `VAE` (Variational Autoencoder) decoding blocks.
- [ ] 186. Convert and validate TF.js `SpeechCommands` audio classifier.
- [ ] 187. Validate 1D Convolution outputs on raw audio sequences.
- [ ] 188. Check 2D Convolution on Mel-spectrogram input formats.
- [ ] 189. Validate UpSampling/Conv2DTranspose artifacts match TF.js completely.

### Phase 14: Subgraphs, Custom Layers & Control Flow

- [ ] 190. Handle Keras `Lambda` layers. (Provide clear errors if un-translatable Python code is found, skip if JS equivalents exist).
- [ ] 191. Attempt to trace JS closures in TF.js GraphModels and map them to ONNX subgraphs.
- [ ] 192. Parse TF.js `ControlFlow` ops (`Switch`, `Merge`, `Enter`, `Exit`, `NextIteration`).
- [ ] 193. Map TF.js `Loop` constructs to ONNX `Loop`.
- [ ] 194. Map TF.js `Cond` constructs to ONNX `If`.
- [ ] 195. Implement a registry for users to inject custom JS converters for their proprietary layers.
- [ ] 196. Provide `registerConverter('MyCustomLayer', (node, builder) => { ... })` API.
- [ ] 197. Handle sub-models (Keras models nested within Keras models) correctly by flattening the graph.
- [ ] 198. Extract `keras_version` and `backend` information and embed into ONNX `producer_name`.

### Phase 15: Browser API, UI, and Packaging

- [ ] 199. Expose TypeScript library API: `const onnxBytes = await keras2onnx(modelJson, weightsBin)`.
- [ ] 200. Build a Node.js CLI: `onnx9000 keras convert my_model.h5 --output my_model.onnx`.
- [ ] 201. Support CLI format auto-detection (inferring TF.js vs HDF5 from file signatures).
- [ ] 202. Build the visual drag-and-drop web converter interface.
- [ ] 203. Provide real-time conversion progress callbacks for UI updates.
- [ ] 204. Handle memory-efficient ArrayBuffer transfers using JS `Transferable` objects.
- [ ] 205. Validate final generated ONNX Protobuf structure using the internal `onnx9000` linting tool.
- [ ] 206. Export the converter logic as an isolated NPM package `@onnx9000/tfjs-converter`.
- [ ] 207. Create an automated migration script mapping standard `tfjs-converter` args to `onnx9000`.

### Phase 16: Optimizations & Graph Rewriting

- [ ] 208. Implement TF.js explicit `FusedBatchNorm` un-fusing if targeting low ONNX opsets.
- [ ] 209. Map TF.js `_FusedConv2D` explicitly to ONNX `Conv` + `Relu` or `Conv` + `Bias` + `Relu`.
- [ ] 210. Map TF.js `_FusedMatMul` to ONNX explicitly.
- [ ] 211. Remove TF.js explicit `Identity` chains injected by SavedModel builders.
- [ ] 212. Resolve static subgraphs (e.g., `Shape` -> `Slice` -> `Concat`) into static initializers to minimize ONNX payload.
- [ ] 213. Rewrite explicit channel transposes out of the graph by swapping weight dimension ordering on standard ops.
- [ ] 214. Clean up `StopGradient` nodes (removing them entirely as ONNX is inference-only).

### Phase 17: Precision & Quantization

- [ ] 215. Parse TF.js `float16` weights natively (handling DataView buffers correctly in JS).
- [ ] 216. Parse TF.js `uint8` quantized weights natively.
- [ ] 217. Read TF.js quantization scale/min/max metadata and embed into ONNX `DequantizeLinear`.
- [ ] 218. Provide an option to cast all weights to `float16` during conversion to save space (`--fp16`).
- [ ] 219. Provide an option to perform W8A16 or W4A16 quantization immediately during conversion.
- [ ] 220. Ensure `int64` tensors in TF.js are gracefully downcast to `int32` for better WebGPU support down the line.

### Phase 18: Ecosystem Parity & Interoperability

- [ ] 221. Establish CI pipeline matching official `tf2onnx` regression tests.
- [ ] 222. Maintain exact equivalence with `tf2onnx` opset 13-19 standards.
- [ ] 223. Convert HuggingFace standard Keras/TF models dynamically via Hub URLs.
- [ ] 224. Support reading `.pb` (Protobuf) TensorFlow SavedModels via a WASM flatbuffer parser.
- [ ] 225. Support parsing TensorFlow Hub URLs directly (`https://tfhub.dev/...`).
- [ ] 226. Produce ONNX models that perfectly load into standard Python `onnxruntime` (not just `onnx9000`).
- [ ] 227. Export a `metadata.json` sidecar preserving Keras training history, class labels, and dictionaries.

### Phase 19: Edge Cases, Quirks, and Telemetry

- [ ] 228. Detect and warn on Keras `input_shape` missing dimensions (e.g., completely dynamic models without defined ranks).
- [ ] 229. Handle `SpaceToBatchND` and `BatchToSpaceND` operations efficiently (often used in dilated convs).
- [ ] 230. Map TF.js `NonMaxSuppressionV3/V4/V5` to ONNX `NonMaxSuppression`.
- [ ] 231. Translate YOLO-specific custom darknet layers if represented in TF.js format.
- [ ] 232. Handle unsupported opcodes by creating custom domains `ai.onnx.contrib.tfjs`.
- [ ] 233. Provide clear error diagnostics displaying the exact node/layer that failed conversion.
- [ ] 234. Avoid "Maximum Call Stack Size Exceeded" when traversing TF.js graphs with 10,000+ nodes.
- [ ] 235. Track specific operator translation failures and report aggregate telemetry.

### Phase 20: Documentation & Final Delivery

- [ ] 236. Create tutorial: "Migrating from TensorFlow.js to WebGPU ONNX in 5 minutes".
- [ ] 237. Create tutorial: "Converting Keras H5 models directly in the Browser".
- [ ] 238. Write detailed API specs for the TS conversion hooks.
- [ ] 239. Include a compatibility matrix mapping Keras Layer versions to supported ONNX opsets.
- [ ] 240. Publish an interactive CodeSandbox template integrating the converter.
- [ ] 241. Add explicit support for `tf.keras.applications.*` extraction tests.
- [ ] 242. Configure memory bounds checking on Web Worker processes to prevent Out Of Memory crashes.
- [ ] 243. Add support for multiple ONNX domains natively.
- [ ] 244. Implement graph cloning utilities for isolated subgraph testing.
- [ ] 245. Track and propagate ONNX type constraints securely during the AST build.
- [ ] 246. Ensure strict JSON sanitization for malicious TF.js manifests.
- [ ] 247. Validate that dynamic batching (Axis 0) propagates correctly globally.
- [ ] 248. Support explicit conversion targets (e.g., optimizing the resulting ONNX specifically for `webnn`).
- [ ] 249. Embed the `onnx9000` logo and conversion timestamp inside the generated ONNX metadata.
- [ ] 250. Map Keras `.keras` zip structure weights dynamically without unzipping fully to disk (streaming unzip).
- [ ] 251. Validate conversion of `tf.einsum` correctly into explicit ONNX math.
- [ ] 252. Map `tf.complex64` types (used in audio FFT models) cleanly if opset supports it, or split to real/imaginary floats.
- [ ] 253. Translate boolean masking logic (`tf.boolean_mask`) to ONNX explicit `Where` and `NonZero`.
- [ ] 254. Ensure `String` tensors in TF.js (e.g., Universal Sentence Encoder text inputs) map perfectly to ONNX STRING inputs.
- [ ] 255. Map TF.js `StringSplit` and `StringToHashBucket` to explicit ONNX sequence structures or fallback JS ops.
- [ ] 256. Handle `RaggedTensors` (common in modern TF NLP models) by translating to padded dense ONNX tensors dynamically.
- [ ] 257. Verify that multiple input layers with varying types (e.g., Image + String) translate safely.
- [ ] 258. Support `Keras 3.x` backend-agnostic topologies.
- [ ] 259. Map custom loss functions inside `.h5` files into pure ONNX if `--export-training` is specified (future proofing).
- [ ] 260. Release final v1.0 parity matching the Python `keras2onnx` capabilities exactly.
- [ ] 261. Add testing for TF.js specific quantization (Float16) models.
- [ ] 262. Support models with dynamic rank (uncommon but supported in TF.js).
- [ ] 263. Properly handle TF.js `BroadcastTo` with ONNX `Expand`.
- [ ] 264. Properly handle TF.js `TensorArray` operations (used in LSTMs).
- [ ] 265. Implement memory limits for the ArrayBuffer loader.
- [ ] 266. Enable progressive loading of weight shards to show loading bars.
- [ ] 267. Map TF.js `Relu6` specifically to ONNX `Clip`.
- [ ] 268. Handle Keras `SeparableConv1D` correctly (differing from 2D logic).
- [ ] 269. Support extraction of Keras `Masking` layer.
- [ ] 270. Handle TF.js `Unpack` and `Pack` operations.
- [ ] 271. Handle TF.js `Fill` operations safely.
- [ ] 272. Process custom initializers gracefully.
- [ ] 273. Support `LeCunNormal`, `GlorotUniform` extraction accurately.
- [ ] 274. Handle `tf.pad` modes (`CONSTANT`, `REFLECT`, `SYMMETRIC`).
- [ ] 275. Map TF.js `SpaceToDepth` directly.
- [ ] 276. Map TF.js `DepthToSpace` directly.
- [ ] 277. Ensure Keras 3 `EinsumDense` is supported.
- [ ] 278. Add conversion validation for Vision Transformers in TF.js.
- [ ] 279. Add specific tests for TF.js Object Detection API exports.
- [ ] 280. Handle `tf.round` explicitly.
- [ ] 281. Handle `tf.floor` and `tf.ceil`.
- [ ] 282. Convert `tf.clipByValue` to ONNX `Clip`.
- [ ] 283. Map `tf.squaredDifference`.
- [ ] 284. Map `tf.reciprocal`.
- [ ] 285. Map `tf.sign`.
- [ ] 286. Handle `tf.logicalNot`, `tf.logicalAnd`, `tf.logicalOr`.
- [ ] 287. Translate `tf.oneHot` to ONNX `OneHot`.
- [ ] 288. Translate `tf.cumsum` to ONNX `CumSum`.
- [ ] 289. Ensure valid error when processing multi-device TF.js models.
- [ ] 290. Parse training state configurations and strip them from inference payload.
- [ ] 291. Translate `tf.linspace` and `tf.range`.
- [ ] 292. Map `tf.diag` and `tf.diagPart`.
- [ ] 293. Support complex matrices (eigen value/vector ops) fallback.
- [ ] 294. Fully integrate Web Worker thread pooling for Transpose operations.
- [ ] 295. Set up dedicated memory cleanup functions (preventing Blob leak).
- [ ] 296. Publish benchmarking metrics comparing TF.js inference vs ONNX WebGPU inference for the same model.
- [ ] 297. Support conversion to specialized target optimizations via CLI `--optimize`.
- [ ] 298. Validate complete `tf2onnx` CLI parity.
- [ ] 299. Create specific issue templates for failed model conversions.
- [ ] 300. Maintain continuous deployment to `@onnx9000/keras` NPM.
