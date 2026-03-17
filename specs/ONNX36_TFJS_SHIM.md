# ONNX36: TF.js API Shim (WebGPU ONNX Drop-In Replacement)

## Original Project Description

TensorFlow.js (TF.js) is Google's flagship JavaScript ecosystem for machine learning in the browser. It features a vast API surface (`tf.tensor()`, `tf.matMul()`, `tf.loadGraphModel()`) and its own WebGL, WASM, and WebGPU execution backends. However, its architecture is inherently tied to TensorFlow's `GraphDef` semantics, which can lead to bloated memory profiles and sub-optimal shader dispatches when running modern Transformer architectures compared to highly optimized ONNX WebGPU runtimes.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

Instead of forcing developers to rewrite their massive web applications from TF.js to a new ONNX API, `onnx9000.tfjs` provides a **100% drop-in replacement API Shim**.

- **Alias-Driven Architecture:** It exports a global `tf` object that precisely mimics the TensorFlow.js API but routes every single mathematical operation and model loading call directly into `onnx9000.array` (ONNX30) and `onnx9000.keras` (ONNX28) under the hood.
- **Instant WebGPU Upgrades:** A developer can simply change `import * as tf from '@tensorflow/tfjs'` to `import * as tf from 'onnx9000/tfjs-shim'`. Their app logic remains untouched, but their model execution is instantly swapped from TF.js's WebGL backend to `onnx9000`'s state-of-the-art WebGPU ONNX execution engine.
- **Zero-Overhead Translators:** When `tf.loadGraphModel()` is called on a legacy TF.js `model.json`, the shim intercepts the call, compiles the JSON into an ONNX graph completely in-memory, and returns an `onnx9000` executing session disguised as a TF.js `GraphModel` object.

---

## Exhaustive Implementation Checklist

### Phase 1: Core System, Environment & Backend Shims

- [ ] 1. Implement global `tf` namespace object.
- [ ] 2. Implement `tf.setBackend(backendName)` interceptor.
- [ ] 3. Map `tf.setBackend('webgl')` natively to `onnx9000` WebGPU (with fallback to WASM).
- [ ] 4. Map `tf.setBackend('wasm')` natively to `onnx9000` WASM SIMD.
- [ ] 5. Map `tf.setBackend('cpu')` natively to `onnx9000` pure JS/WASM scalar fallback.
- [ ] 6. Implement `tf.getBackend()` returning the active simulated backend.
- [ ] 7. Implement `tf.ready()` resolving a Promise when `onnx9000` WASM/WebGPU is initialized.
- [ ] 8. Implement `tf.env()` configuration stub.
- [ ] 9. Implement `tf.enableProdMode()` (suppressing ONNX compilation warnings).
- [ ] 10. Implement `tf.enableDebugMode()` (enabling ONNX verbose trace logging).
- [ ] 11. Implement `tf.memory()` returning an object matching `{ numBytes, numTensors, numDataBuffers }`.
- [ ] 12. Map `tf.memory()` metrics dynamically to the `onnx9000` internal WebGPU buffer tracker.
- [ ] 13. Implement `tf.profile(f)` wrapping `onnx9000.profile()`.
- [ ] 14. Implement `tf.time(f)` executing the ONNX JIT and measuring wall clock time.
- [ ] 15. Implement `tf.disposeVariables()` clearing the `onnx9000` global context.
- [ ] 16. Emulate TF.js global registry to track active tensors for memory leak detection.
- [ ] 17. Support standard `tf.version.tfjs` matching the latest shimmed version (e.g., `4.10.0`).
- [ ] 18. Support standard `tf.version.core` strings.

### Phase 2: Tensor Creation & Lifecycle Management (`tf.tensor`)

- [ ] 19. Implement `tf.tensor(values, shape, dtype)` mapped to `onnx9000.Tensor`.
- [ ] 20. Implement `tf.tensor1d(values, dtype)`.
- [ ] 21. Implement `tf.tensor2d(values, shape, dtype)`.
- [ ] 22. Implement `tf.tensor3d(values, shape, dtype)`.
- [ ] 23. Implement `tf.tensor4d(values, shape, dtype)`.
- [ ] 24. Implement `tf.tensor5d(values, shape, dtype)`.
- [ ] 25. Implement `tf.tensor6d(values, shape, dtype)`.
- [ ] 26. Implement `tf.scalar(value, dtype)`.
- [ ] 27. Implement `tf.buffer(shape, dtype, values)` mapping to mutable JS arrays.
- [ ] 28. Implement `tf.clone(x)`.
- [ ] 29. Implement `tf.complex(real, imag)` (mapping to `Float32` pairs if ONNX lacks native complex support).
- [ ] 30. Implement `tf.diag(x)`.
- [ ] 31. Implement `tf.eye(numRows, numColumns, batchShape, dtype)`.
- [ ] 32. Implement `tf.fill(shape, value, dtype)`.
- [ ] 33. Implement `tf.imag(complexTensor)`.
- [ ] 34. Implement `tf.linspace(start, stop, num)`.
- [ ] 35. Implement `tf.ones(shape, dtype)`.
- [ ] 36. Implement `tf.onesLike(x)`.
- [ ] 37. Implement `tf.print(x, verbose)` wrapping `console.log(tensor.numpy())`.
- [ ] 38. Implement `tf.range(start, stop, step, dtype)`.
- [ ] 39. Implement `tf.real(complexTensor)`.
- [ ] 40. Implement `tf.zeros(shape, dtype)`.
- [ ] 41. Implement `tf.zerosLike(x)`.

### Phase 3: The `tf.tidy` Memory Engine

- [ ] 42. Implement `tf.tidy(nameOrFn, fn)` scoping block.
- [ ] 43. Track all `onnx9000.Tensor` allocations created inside the `tf.tidy` closure.
- [ ] 44. Prevent intermediate `onnx9000` WebGPU Buffers from leaking out of the closure.
- [ ] 45. Allow the returned `Tensor` (or array of Tensors) to escape the `tidy` block safely.
- [ ] 46. Implement `tf.keep(x)` to exempt a tensor from `tf.tidy` cleanup.
- [ ] 47. Implement `tf.dispose(tensors)` translating to `onnx9000.Tensor.dispose()`.
- [ ] 48. Handle deep arrays and dictionaries of tensors correctly inside `tf.dispose`.
- [ ] 49. Map tensor disposal directly to WebGPU `buffer.destroy()` to ensure VRAM is released immediately.
- [ ] 50. Gracefully catch and ignore double-dispose calls.

### Phase 4: Basic Math & Elementwise Operations

- [ ] 51. Implement `tf.add(a, b)` -> ONNX `Add`.
- [ ] 52. Implement `tf.sub(a, b)` -> ONNX `Sub`.
- [ ] 53. Implement `tf.mul(a, b)` -> ONNX `Mul`.
- [ ] 54. Implement `tf.div(a, b)` -> ONNX `Div`.
- [ ] 55. Implement `tf.divNoNan(a, b)`.
- [ ] 56. Implement `tf.floorDiv(a, b)` -> ONNX `Div` + `Floor`.
- [ ] 57. Implement `tf.maximum(a, b)` -> ONNX `Max`.
- [ ] 58. Implement `tf.minimum(a, b)` -> ONNX `Min`.
- [ ] 59. Implement `tf.mod(a, b)` -> ONNX `Mod`.
- [ ] 60. Implement `tf.pow(base, exp)` -> ONNX `Pow`.
- [ ] 61. Implement `tf.squaredDifference(a, b)` -> ONNX `Sub` + `Pow(2)`.
- [ ] 62. Implement `tf.addN(tensors)` -> ONNX chained `Add`.
- [ ] 63. Implement `tf.abs(x)` -> ONNX `Abs`.
- [ ] 64. Implement `tf.acos(x)` -> ONNX `Acos`.
- [ ] 65. Implement `tf.acosh(x)` -> ONNX `Acosh`.
- [ ] 66. Implement `tf.asin(x)` -> ONNX `Asin`.
- [ ] 67. Implement `tf.asinh(x)` -> ONNX `Asinh`.
- [ ] 68. Implement `tf.atan(x)` -> ONNX `Atan`.
- [ ] 69. Implement `tf.atan2(a, b)` -> ONNX Custom/Math.
- [ ] 70. Implement `tf.atanh(x)` -> ONNX `Atanh`.
- [ ] 71. Implement `tf.ceil(x)` -> ONNX `Ceil`.
- [ ] 72. Implement `tf.cos(x)` -> ONNX `Cos`.
- [ ] 73. Implement `tf.cosh(x)` -> ONNX `Cosh`.
- [ ] 74. Implement `tf.erf(x)` -> ONNX `Erf`.
- [ ] 75. Implement `tf.exp(x)` -> ONNX `Exp`.
- [ ] 76. Implement `tf.expm1(x)` -> ONNX `Exp` + `Sub(1)`.
- [ ] 77. Implement `tf.floor(x)` -> ONNX `Floor`.
- [ ] 78. Implement `tf.isFinite(x)`.
- [ ] 79. Implement `tf.isInf(x)` -> ONNX `IsInf`.
- [ ] 80. Implement `tf.isNaN(x)` -> ONNX `IsNaN`.
- [ ] 81. Implement `tf.log(x)` -> ONNX `Log`.
- [ ] 82. Implement `tf.log1p(x)` -> ONNX `Add(1)` + `Log`.
- [ ] 83. Implement `tf.neg(x)` -> ONNX `Neg`.
- [ ] 84. Implement `tf.reciprocal(x)` -> ONNX `Reciprocal`.
- [ ] 85. Implement `tf.round(x)` -> ONNX `Round`.
- [ ] 86. Implement `tf.rsqrt(x)` -> ONNX `Sqrt` + `Reciprocal`.
- [ ] 87. Implement `tf.sign(x)` -> ONNX `Sign`.
- [ ] 88. Implement `tf.sin(x)` -> ONNX `Sin`.
- [ ] 89. Implement `tf.sinh(x)` -> ONNX `Sinh`.
- [ ] 90. Implement `tf.sqrt(x)` -> ONNX `Sqrt`.
- [ ] 91. Implement `tf.square(x)` -> ONNX `Pow(2)`.
- [ ] 92. Implement `tf.tan(x)` -> ONNX `Tan`.
- [ ] 93. Implement `tf.step(x, alpha)` -> ONNX `Where`.

### Phase 5: Matrix Algebra & Convolutions

- [ ] 94. Implement `tf.matMul(a, b, transposeA, transposeB)` -> ONNX `MatMul` (with transposition hooks).
- [ ] 95. Implement `tf.dot(a, b)`.
- [ ] 96. Implement `tf.norm(x, ord, axis, keepDims)`.
- [ ] 97. Implement `tf.outerProduct(v1, v2)`.
- [ ] 98. Implement `tf.conv1d(x, filter, stride, pad, dataFormat, dilation)`.
- [ ] 99. Translate TF.js `padding='same'` / `'valid'` explicitly to ONNX spatial paddings.
- [ ] 100. Handle `dataFormat` mapping (NHWC vs NCHW) securely injecting ONNX `Transpose` if required.
- [ ] 101. Implement `tf.conv2d(x, filter, strides, pad, dataFormat, dilations)`.
- [ ] 102. Implement `tf.conv3d(x, filter, strides, pad, dataFormat, dilations)`.
- [ ] 103. Implement `tf.depthwiseConv2d(x, filter, strides, pad, dataFormat, dilations)`.
- [ ] 104. Map `DepthwiseConv2D` strictly to ONNX `Conv` with `group` parameter adjustments.
- [ ] 105. Implement `tf.separableConv2d(x, depthwiseFilter, pointwiseFilter, strides, pad, dilation, dataFormat)`.
- [ ] 106. Implement `tf.conv2dTranspose(x, filter, outputShape, strides, pad)`.
- [ ] 107. Implement `tf.conv3dTranspose(x, filter, outputShape, strides, pad)`.

### Phase 6: Reductions & Pooling

- [ ] 108. Implement `tf.argMax(x, axis)` -> ONNX `ArgMax`.
- [ ] 109. Implement `tf.argMin(x, axis)` -> ONNX `ArgMin`.
- [ ] 110. Implement `tf.max(x, axis, keepDims)` -> ONNX `ReduceMax`.
- [ ] 111. Implement `tf.mean(x, axis, keepDims)` -> ONNX `ReduceMean`.
- [ ] 112. Implement `tf.min(x, axis, keepDims)` -> ONNX `ReduceMin`.
- [ ] 113. Implement `tf.prod(x, axis, keepDims)` -> ONNX `ReduceProd`.
- [ ] 114. Implement `tf.sum(x, axis, keepDims)` -> ONNX `ReduceSum`.
- [ ] 115. Implement `tf.all(x, axis, keepDims)`.
- [ ] 116. Implement `tf.any(x, axis, keepDims)`.
- [ ] 117. Implement `tf.logSumExp(x, axis, keepDims)`.
- [ ] 118. Implement `tf.maxPool(x, filterSize, strides, pad, dimRoundingMode)`.
- [ ] 119. Implement `tf.avgPool(x, filterSize, strides, pad, dimRoundingMode)`.
- [ ] 120. Implement `tf.maxPool3d()`.
- [ ] 121. Implement `tf.avgPool3d()`.
- [ ] 122. Implement `tf.pool(input, windowShape, poolingType, pad, dilations, strides)`.

### Phase 7: Tensor Manipulation, Slicing & Routing

- [ ] 123. Implement `tf.cast(x, dtype)` -> ONNX `Cast`.
- [ ] 124. Implement `tf.expandDims(x, axis)` -> ONNX `Unsqueeze`.
- [ ] 125. Implement `tf.squeeze(x, axis)` -> ONNX `Squeeze`.
- [ ] 126. Implement `tf.reshape(x, shape)` -> ONNX `Reshape`.
- [ ] 127. Implement `tf.transpose(x, perm)` -> ONNX `Transpose`.
- [ ] 128. Implement `tf.concat(tensors, axis)` -> ONNX `Concat`.
- [ ] 129. Implement `tf.split(x, numOrSizeSplits, axis)` -> ONNX `Split`.
- [ ] 130. Implement `tf.stack(tensors, axis)`.
- [ ] 131. Implement `tf.unstack(x, axis)`.
- [ ] 132. Implement `tf.pad(x, paddings, constantValue)` -> ONNX `Pad`.
- [ ] 133. Implement `tf.pad1d()`, `tf.pad2d()`, `tf.pad3d()`, `tf.pad4d()`.
- [ ] 134. Implement `tf.slice(x, begin, size)` -> ONNX `Slice`.
- [ ] 135. Implement `tf.slice1d()`, `tf.slice2d()`, `tf.slice3d()`, `tf.slice4d()`.
- [ ] 136. Implement `tf.stridedSlice(x, begin, end, strides, beginMask, endMask...)`.
- [ ] 137. Convert `stridedSlice` bitmasks correctly into explicit ONNX start/end coordinates dynamically.
- [ ] 138. Implement `tf.gather(x, indices, axis)` -> ONNX `Gather`.
- [ ] 139. Implement `tf.gatherND(x, indices)` -> ONNX `GatherND`.
- [ ] 140. Implement `tf.scatterND(indices, updates, shape)` -> ONNX `ScatterND` (emulation using ConstantOfShape + ScatterND).
- [ ] 141. Implement `tf.tensorScatterUpdate(tensor, indices, updates)` -> ONNX `ScatterND`.
- [ ] 142. Implement `tf.booleanMaskAsync(tensor, mask, axis)` -> ONNX `NonZero` + `Gather`.
- [ ] 143. Implement `tf.whereAsync(condition)` -> ONNX `NonZero`.
- [ ] 144. Implement `tf.reverse(x, axis)` -> ONNX `ReverseSequence`.
- [ ] 145. Implement `tf.reverse1d()`, `tf.reverse2d()`, etc.
- [ ] 146. Implement `tf.tile(x, reps)` -> ONNX `Tile`.
- [ ] 147. Implement `tf.spaceToBatchND(x, blockShape, paddings)`.
- [ ] 148. Implement `tf.batchToSpaceND(x, blockShape, crops)`.
- [ ] 149. Implement `tf.depthToSpace(x, blockSize, dataFormat)`.
- [ ] 150. Implement `tf.spaceToDepth(x, blockSize, dataFormat)`.

### Phase 8: Logical, Relational & Boolean Operations

- [ ] 151. Implement `tf.equal(a, b)` -> ONNX `Equal`.
- [ ] 152. Implement `tf.notEqual(a, b)`.
- [ ] 153. Implement `tf.less(a, b)` -> ONNX `Less`.
- [ ] 154. Implement `tf.lessEqual(a, b)` -> ONNX `LessOrEqual`.
- [ ] 155. Implement `tf.greater(a, b)` -> ONNX `Greater`.
- [ ] 156. Implement `tf.greaterEqual(a, b)` -> ONNX `GreaterOrEqual`.
- [ ] 157. Implement `tf.logicalAnd(a, b)` -> ONNX `And`.
- [ ] 158. Implement `tf.logicalOr(a, b)` -> ONNX `Or`.
- [ ] 159. Implement `tf.logicalNot(x)` -> ONNX `Not`.
- [ ] 160. Implement `tf.logicalXor(a, b)` -> ONNX `Xor`.
- [ ] 161. Implement `tf.where(condition, a, b)` -> ONNX `Where`.

### Phase 9: Activations & Neural Network Core (`tf.nn`)

- [ ] 162. Implement `tf.relu(x)` -> ONNX `Relu`.
- [ ] 163. Implement `tf.relu6(x)` -> ONNX `Clip`.
- [ ] 164. Implement `tf.leakyRelu(x, alpha)` -> ONNX `LeakyRelu`.
- [ ] 165. Implement `tf.elu(x)` -> ONNX `Elu`.
- [ ] 166. Implement `tf.selu(x)` -> ONNX `Selu`.
- [ ] 167. Implement `tf.sigmoid(x)` -> ONNX `Sigmoid`.
- [ ] 168. Implement `tf.softmax(x, axis)` -> ONNX `Softmax`.
- [ ] 169. Implement `tf.logSoftmax(x, axis)` -> ONNX `LogSoftmax`.
- [ ] 170. Implement `tf.softplus(x)` -> ONNX `Softplus`.
- [ ] 171. Implement `tf.step(x, alpha)`.
- [ ] 172. Implement `tf.localResponseNormalization(x, depthRadius, bias, alpha, beta)`.

### Phase 10: Model Loading & Graph Execution (`tf.loadGraphModel`)

- [ ] 173. Implement `tf.loadGraphModel(modelUrl, options)` interceptor.
- [ ] 174. Download `model.json` and weight shards natively inside the shim.
- [ ] 175. Route the downloaded TF.js GraphDef through `onnx9000.keras` to generate an ONNX AST entirely in memory.
- [ ] 176. Instantiate an `onnx9000` execution session (WebGPU/WASM) wrapped in a mock `tf.GraphModel` object.
- [ ] 177. Implement `model.predict(inputs)` on the mocked `GraphModel`.
- [ ] 178. Implement `model.execute(inputs)` on the mocked `GraphModel`.
- [ ] 179. Implement `model.executeAsync(inputs)` returning promises securely.
- [ ] 180. Translate passed TF.js Tensors (from the caller) automatically into ONNX buffer formats before execution.
- [ ] 181. Translate returned ONNX Tensors back into mock `tf.Tensor` objects before returning to the caller.
- [ ] 182. Implement `model.inputs` property matching TF.js metadata structs.
- [ ] 183. Implement `model.outputs` property matching TF.js metadata structs.
- [ ] 184. Implement `model.weights` dictionary (returning constants if accessed explicitly).
- [ ] 185. Provide `model.dispose()` mapping to ONNX session destruction.
- [ ] 186. Handle ONNX-specific dynamic batch limits seamlessly so legacy TF.js `.predict()` calls don't crash.
- [ ] 187. Implement `tf.loadLayersModel(modelUrl, options)` interceptor.
- [ ] 188. Route HDF5 / Keras definitions to `onnx9000` execution engine identically to GraphModels.

### Phase 11: Web Image, Video & Data Utilities (`tf.browser`)

- [ ] 189. Implement `tf.browser.fromPixels(pixels, numChannels)` mapping `ImageData`, `HTMLVideoElement`, or `HTMLImageElement` to ONNX arrays.
- [ ] 190. Execute zero-copy WebGPU buffer mappings for `fromPixels` using `createImageBitmap` natively.
- [ ] 191. Implement `tf.browser.toPixels(tensor, canvas)` mapping ONNX buffers back to a Canvas `ImageData` object.
- [ ] 192. Ensure async execution of `.toPixels()` maintains UI responsiveness.
- [ ] 193. Implement `tf.image.resizeBilinear(images, size, alignCorners, halfPixelCenters)`.
- [ ] 194. Implement `tf.image.resizeNearestNeighbor(images, size, alignCorners, halfPixelCenters)`.
- [ ] 195. Implement `tf.image.cropAndResize(image, boxes, boxInd, cropSize, method, extrapolationValue)`.
- [ ] 196. Implement `tf.image.nonMaxSuppression(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold)`.
- [ ] 197. Map `nonMaxSuppression` specifically to ONNX `NonMaxSuppression` or WASM fast paths if ONNX execution is too heavy for NMS.
- [ ] 198. Implement `tf.image.nonMaxSuppressionAsync()`.
- [ ] 199. Implement `tf.image.nonMaxSuppressionWithScore()`.
- [ ] 200. Implement `tf.image.flipLeftRight(image)`.

### Phase 12: Machine Learning Layers API (`tf.layers`)

- [ ] 201. Define `tf.layers` namespace object.
- [ ] 202. Implement `tf.sequential(config)` builder returning a mock Model.
- [ ] 203. Implement `tf.model(config)` builder returning a functional mock Model.
- [ ] 204. Implement `model.add(layer)` logic.
- [ ] 205. Implement `tf.layers.dense(config)` mapping to an ONNX subgraph generator.
- [ ] 206. Implement `tf.layers.conv2d(config)`.
- [ ] 207. Implement `tf.layers.maxPooling2d(config)`.
- [ ] 208. Implement `tf.layers.flatten(config)`.
- [ ] 209. Implement `tf.layers.dropout(config)`.
- [ ] 210. Implement `tf.layers.batchNormalization(config)`.
- [ ] 211. Implement `tf.layers.reLU(config)`.
- [ ] 212. Ensure `model.compile()` functions flawlessly (even if acting as a no-op stub for inference-only environments).
- [ ] 213. Ensure `model.predict()` evaluates the dynamically built `tf.layers` graph by compiling it instantly to ONNX and running it.
- [ ] 214. Handle complex `tf.layers` merging (e.g., `tf.layers.add()`, `tf.layers.concatenate()`).
- [ ] 215. Implement layer weight extraction (`layer.getWeights()`).
- [ ] 216. Implement layer weight setting (`layer.setWeights(weights)`).

### Phase 13: Operations Execution Control (Eager vs Graph)

- [ ] 217. Identify Eager vs Lazy invocation dynamically. If a user calls `tf.add(a, b)` where `a` and `b` are real data, execute the ONNX math kernel instantly.
- [ ] 218. Identify Symbolic invocation. If a user calls `tf.add(a, b)` inside a `tf.model()` topology build, emit ONNX AST nodes instead of executing.
- [ ] 219. Maintain strict API parity with the TF.js `SymbolicTensor` vs `Tensor` distinction.
- [ ] 220. Automatically cast JavaScript native nested arrays `[[1, 2], [3, 4]]` passed to math functions into ONNX tensors.
- [ ] 221. Support standard Promise-based `.data()` extraction (`await tensor.data()`).
- [ ] 222. Support synchronous `.dataSync()` extraction (throwing explicit errors if running in WebGPU where sync extraction is forbidden).
- [ ] 223. Implement `.array()` returning nested JS arrays.
- [ ] 224. Implement `.arraySync()` returning nested JS arrays.
- [ ] 225. Ensure TF.js unique prototype methods (e.g., `tensor.flatten()`) map to the correct global namespace functions.

### Phase 14: End-to-End Validation (Replacing standard TF.js Apps)

- [ ] 226. Unit Test: Load standard `@tensorflow-models/posenet` NPM package utilizing the `onnx9000` shim and verify flawless webcam execution.
- [ ] 227. Unit Test: Load standard `@tensorflow-models/body-pix`.
- [ ] 228. Unit Test: Load standard `@tensorflow-models/blazeface`.
- [ ] 229. Unit Test: Load standard `@tensorflow-models/universal-sentence-encoder` (USE).
- [ ] 230. Unit Test: Load standard `@tensorflow-models/coco-ssd`.
- [ ] 231. Unit Test: Load standard `@tensorflow-models/mobilenet`.
- [ ] 232. Verify memory limits do not exceed original TF.js limits during extended execution loops (e.g., running PoseNet on requestAnimationFrame).
- [ ] 233. Measure FPS improvements visually when migrating from TF.js WebGL to `onnx9000` WebGPU.
- [ ] 234. Verify `npm install @tensorflow/tfjs` can be successfully aliased in `package.json` or Webpack/Vite `resolve.alias` configs to point to the shim.

### Phase 15: Autodiff, Gradients & Training Stubs

- [ ] 235. Implement `tf.variable(initialValue, trainable, name, dtype)`.
- [ ] 236. Implement `tf.grad(f)`.
- [ ] 237. Implement `tf.grads(f)`.
- [ ] 238. Implement `tf.valueAndGrad(f)`.
- [ ] 239. Implement `tf.customGrad(f)`.
- [ ] 240. Implement `tf.train.sgd(learningRate)`.
- [ ] 241. Implement `tf.train.adam(learningRate, beta1, beta2, epsilon)`.
- [ ] 242. Support `.applyGradients()` execution.
- [ ] 243. Connect these gradient functions natively to `onnx9000.training`'s AST-based autograd engine if available.
- [ ] 244. If training is not configured, provide highly descriptive stubs explaining that the shim is optimized for inference.

### Phase 16: Error Mapping & Debugging Consistency

- [ ] 245. Map `onnx9000` dimension mismatch exceptions exactly to the standard TF.js `Error: Incompatible shapes: [x,y] vs. [a,b]` text format.
- [ ] 246. Mimic TF.js console warnings if an operation forces a slow CPU readback.
- [ ] 247. Support passing string configurations into `tf.cast` securely.
- [ ] 248. Provide an API to list uniquely executed Kernels for developer debugging.
- [ ] 249. Replicate `tf.print()` formatting (with specific decimal truncations and alignment).
- [ ] 250. Ensure custom user extensions wrapping the `tf` object properties do not crash.

### Phase 17: String Tensors & NLP Edge Cases

- [ ] 251. Handle `dtype='string'` natively since TF.js uses string tensors extensively in NLP pipelines (USE, BERT).
- [ ] 252. Map `tf.string` values to `onnx9000` String arrays correctly.
- [ ] 253. Implement `tf.string.stringSplit()`.
- [ ] 254. Implement `tf.string.stringToHashBucketFast()`.
- [ ] 255. Map string hashing specifically to the ONNX equivalent operations if the standard TF.js NLP models use them.

### Phase 18: Random Number Generation

- [ ] 256. Implement `tf.randomUniform(shape, minval, maxval, dtype, seed)`.
- [ ] 257. Implement `tf.randomNormal(shape, mean, stdDev, dtype, seed)`.
- [ ] 258. Implement `tf.truncatedNormal(shape, mean, stdDev, dtype, seed)`.
- [ ] 259. Implement `tf.randomGamma(shape, alpha, beta, dtype, seed)`.
- [ ] 260. Implement `tf.multinomial(logits, numSamples, seed, normalized)`.
- [ ] 261. Guarantee reproducible random seeds identical to TF.js's underlying LCG (Linear Congruential Generator) implementation.

### Phase 19: Edge Cases, Quirks, and Compatibility Options

- [ ] 262. Support `.clipByValue(min, max)` mapped to ONNX `Clip`.
- [ ] 263. Support `.pad(paddings, constantValue)`.
- [ ] 264. Support `tf.setDevice()` routing (e.g., mapping to specific WebGPU adapters).
- [ ] 265. Emulate `tf.memory()` exact string structures expected by TF.js developer tooling.
- [ ] 266. Support `tf.nextFrame()` yielding to the browser event loop natively.
- [ ] 267. Map `tf.util.encodeString` and `tf.util.decodeString` properly.
- [ ] 268. Provide `tf.util.fetch` mapping to standard window fetch logic.
- [ ] 269. Extract `strides` configurations accurately (translating single numbers to array tuples internally).
- [ ] 270. Handle zero-sized tensors flawlessly (crucial for TF.js control flow operations).

### Phase 20: Delivery & Documentation

- [ ] 271. Publish to NPM under a specific alias `@onnx9000/tfjs-shim`.
- [ ] 272. Provide Webpack configuration snippets demonstrating how to alias `@tensorflow/tfjs` imports dynamically to the shim during build-time.
- [ ] 273. Provide Vite configuration snippets for dynamic aliasing.
- [ ] 274. Create benchmark reports explicitly showcasing frame-rate increases on classic TF.js web applications (e.g., MediaPipe/PoseNet).
- [ ] 275. Ensure TypeScript declarations (`.d.ts`) perfectly match `@tensorflow/tfjs/dist/index.d.ts` to prevent IDE compiler errors.
- [ ] 276. Write a migration guide: "Upgrading your TF.js application to WebGPU ONNX with Zero Code Changes".
- [ ] 277. Validate execution natively in React Native via `tfjs-react-native` polyfill bridging.
- [ ] 278. Establish continuous integration comparing the output of the shim directly against a live headless instance running genuine TF.js.
- [ ] 279. Maintain an ongoing compatibility matrix tracking unsupported esoteric `tf.*` operations.
- [ ] 280. Handle `tf.Einsum` execution.
- [ ] 281. Handle `tf.cumprod` execution.
- [ ] 282. Handle `tf.cumsum` execution.
- [ ] 283. Support specific `tf.losses.*` and `tf.metrics.*` modules (or return raw functions).
- [ ] 284. Map `tf.io.browserFiles` and `tf.io.browserHTTPRequest` precisely to standard local file loaders.
- [ ] 285. Support mapping `tf.tensor1d` directly to optimized TypedArrays.
- [ ] 286. Handle specific 1D dimensional expansions in mathematical broadcasting.
- [ ] 287. Recreate `tf.signal.stft` processing exactly to map standard Audio execution tasks.
- [ ] 288. Simulate specific TF.js `NaN` mask rules.
- [ ] 289. Emulate `tf.spectral.rfft` natively or via WASM loops if ONNX support is missing.
- [ ] 290. Extract string values safely out of `.data()` promises.
- [ ] 291. Manage ArrayBuffer Detachment explicitly upon tensor disposal.
- [ ] 292. Add support for creating a Web Worker dedicated specifically to the TF.js Eager evaluations.
- [ ] 293. Build interactive examples demonstrating the exact same React code running on TF.js and the Shim simultaneously.
- [ ] 294. Validate memory leak absence in 1,000,000+ operation loops.
- [ ] 295. Configure explicit fallback logic for unsupported `WebGL2` specific functions if they exist.
- [ ] 296. Validate execution cleanly in Node.js (replacing `@tensorflow/tfjs-node`).
- [ ] 297. Support conversion directly to `onnx9000.genai` outputs.
- [ ] 298. Validate precise execution under explicit memory bounds checking on mobile Safari.
- [ ] 299. Write comprehensive API documentation mapping TF.js to ONNX AST.
- [ ] 300. Release v1.0 feature complete certification for `onnx9000.tfjs-shim` achieving full parity with TF.js Core.
