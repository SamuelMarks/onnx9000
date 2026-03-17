# ONNX30: onnx-array-api (Web-Native NumPy/Eager API for ONNX)

## Original Project Description

`onnx-array-api` is a Python library that provides a NumPy-like, Eager-execution API for dynamically creating and evaluating ONNX graphs. Instead of writing verbose `onnx.helper` node definitions or tracing a PyTorch model, developers can write mathematical operations using standard array semantics (e.g., `z = x + y * 2`), and the library automatically constructs the corresponding ONNX graph and executes it via ONNX Runtime under the hood. It bridges the gap between static graph definition and eager, interactive numerical computing.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

`onnx9000.array` provides this exact NumPy-like experience, but entirely within JavaScript/TypeScript and Pyodide, with zero dependency on the C++ ONNX Runtime.

- **Dual Language Support:** Provides both a Python API (for Pyodide/JupyterLite) and a native TypeScript API (for browser-based apps), sharing the exact same underlying WASM math kernels.
- **Lazy vs Eager Toggling:** Operations can either execute instantly via WebGPU/WASM (Eager mode) or build up a massive ONNX `GraphProto` in the background (Lazy mode) to be exported as a `.onnx` file later.
- **No Python Required for TS Developers:** JavaScript developers get a full NumPy/TensorFlow.js-like numerical library (`import * as np from 'onnx9000/array'`) that natively speaks ONNX Protobuf, making it trivial to author ONNX models strictly via JS math syntax.
- **JIT Compilation to WGSL:** In Lazy mode, complex mathematical expressions written in JS are JIT-compiled directly into fused WebGPU WGSL shaders rather than executing node-by-node.

---

## Exhaustive Implementation Checklist

### Phase 1: Core Array/Tensor Object (`onnx9000.array.Tensor`)

- [ ] 1. Define the base `EagerTensor` class extending `onnx9000.Tensor`.
- [ ] 2. Define the `LazyTensor` class (stores an AST node reference rather than raw data).
- [ ] 3. Implement `Tensor` instantiation from JS Arrays (`const t = np.array([1, 2, 3])`).
- [ ] 4. Implement instantiation from TypedArrays (`Float32Array`, `Int32Array`).
- [ ] 5. Implement explicit `dtype` forcing during instantiation (`dtype='float16'`).
- [ ] 6. Implement `Tensor.shape` property getter.
- [ ] 7. Implement `Tensor.dtype` property getter.
- [ ] 8. Implement `Tensor.ndim` property getter.
- [ ] 9. Implement `Tensor.size` property getter.
- [ ] 10. Implement `Tensor.numpy()` / `Tensor.data()` to extract raw values.
- [ ] 11. Support printing tensors gracefully to the console (truncating large arrays).
- [ ] 12. Implement `np.zeros(shape, dtype)`.
- [ ] 13. Implement `np.ones(shape, dtype)`.
- [ ] 14. Implement `np.empty(shape, dtype)`.
- [ ] 15. Implement `np.full(shape, fill_value, dtype)`.
- [ ] 16. Implement `np.eye(N, M, k, dtype)`.
- [ ] 17. Implement `np.identity(n, dtype)`.
- [ ] 18. Implement `np.arange(start, stop, step, dtype)`.
- [ ] 19. Implement `np.linspace(start, stop, num, endpoint, dtype)`.
- [ ] 20. Implement automatic context management (switching between Lazy builder mode and Eager mode).

### Phase 2: Basic Mathematical Operations (Eager & Lazy)

- [ ] 21. Implement `add(a, b)` / `a.add(b)`.
- [ ] 22. Implement `subtract(a, b)` / `a.sub(b)`.
- [ ] 23. Implement `multiply(a, b)` / `a.mul(b)`.
- [ ] 24. Implement `divide(a, b)` / `a.div(b)`.
- [ ] 25. Implement `power(a, b)` / `a.pow(b)`.
- [ ] 26. Implement `mod(a, b)`.
- [ ] 27. Implement `absolute(a)` / `a.abs()`.
- [ ] 28. Implement `negative(a)` / `a.neg()`.
- [ ] 29. Implement `sign(a)`.
- [ ] 30. Implement `exp(a)`.
- [ ] 31. Implement `log(a)`.
- [ ] 32. Implement `log10(a)`.
- [ ] 33. Implement `log2(a)`.
- [ ] 34. Implement `sqrt(a)`.
- [ ] 35. Implement `square(a)`.
- [ ] 36. Implement `cbrt(a)`.
- [ ] 37. Implement `reciprocal(a)`.
- [ ] 38. Support implicit scalar-to-tensor broadcasting in all math ops (e.g., `a.add(5)`).
- [ ] 39. Support standard NumPy broadcasting rules (e.g., `[3, 1] + [1, 4] -> [3, 4]`).
- [ ] 40. Ensure Lazy mode emits `Constant` nodes automatically for scalar arguments.

### Phase 3: Trigonometric Operations

- [ ] 41. Implement `sin(a)`.
- [ ] 42. Implement `cos(a)`.
- [ ] 43. Implement `tan(a)`.
- [ ] 44. Implement `arcsin(a)`.
- [ ] 45. Implement `arccos(a)`.
- [ ] 46. Implement `arctan(a)`.
- [ ] 47. Implement `sinh(a)`.
- [ ] 48. Implement `cosh(a)`.
- [ ] 49. Implement `tanh(a)`.
- [ ] 50. Implement `arcsinh(a)`.
- [ ] 51. Implement `arccosh(a)`.
- [ ] 52. Implement `arctanh(a)`.
- [ ] 53. Implement `deg2rad(a)`.
- [ ] 54. Implement `rad2deg(a)`.

### Phase 4: Matrix & Linear Algebra Operations

- [ ] 55. Implement `matmul(a, b)`.
- [ ] 56. Implement `dot(a, b)`.
- [ ] 57. Implement `vdot(a, b)`.
- [ ] 58. Implement `inner(a, b)`.
- [ ] 59. Implement `outer(a, b)`.
- [ ] 60. Implement `tensordot(a, b, axes)`.
- [ ] 61. Implement `einsum(subscripts, ...operands)`.
- [ ] 62. Implement `transpose(a, axes)`.
- [ ] 63. Implement `a.T` shorthand for matrix transposition.
- [ ] 64. Implement `swapaxes(a, axis1, axis2)`.
- [ ] 65. Implement `trace(a, offset, axis1, axis2)`.
- [ ] 66. Implement `linalg.norm(x, ord, axis, keepdims)`.
- [ ] 67. Implement `linalg.det(a)`.
- [ ] 68. Implement `linalg.inv(a)` (using WASM fallbacks if ONNX natively lacks it).
- [ ] 69. Implement `linalg.solve(a, b)`.
- [ ] 70. Map complex linear algebra natively to ONNX loops or WebGPU compute if standard ops are insufficient.

### Phase 5: Reduction Operations

- [ ] 71. Implement `sum(a, axis, keepdims)`.
- [ ] 72. Implement `prod(a, axis, keepdims)`.
- [ ] 73. Implement `mean(a, axis, keepdims)`.
- [ ] 74. Implement `std(a, axis, keepdims)`.
- [ ] 75. Implement `var(a, axis, keepdims)`.
- [ ] 76. Implement `min(a, axis, keepdims)`.
- [ ] 77. Implement `max(a, axis, keepdims)`.
- [ ] 78. Implement `argmin(a, axis, keepdims)`.
- [ ] 79. Implement `argmax(a, axis, keepdims)`.
- [ ] 80. Implement `ptp(a, axis)` (Peak to Peak - Max minus Min).
- [ ] 81. Implement `all(a, axis, keepdims)`.
- [ ] 82. Implement `any(a, axis, keepdims)`.
- [ ] 83. Implement `cumsum(a, axis)`.
- [ ] 84. Implement `cumprod(a, axis)`.

### Phase 6: Shape Manipulation & Array Operations

- [ ] 85. Implement `reshape(a, newshape)`.
- [ ] 86. Implement `a.reshape(newshape)`.
- [ ] 87. Implement `ravel(a)` (flattening to 1D).
- [ ] 88. Implement `squeeze(a, axis)`.
- [ ] 89. Implement `expand_dims(a, axis)`.
- [ ] 90. Implement `broadcast_to(array, shape)`.
- [ ] 91. Implement `concatenate(arrays, axis)`.
- [ ] 92. Implement `stack(arrays, axis)`.
- [ ] 93. Implement `vstack(tup)`.
- [ ] 94. Implement `hstack(tup)`.
- [ ] 95. Implement `dstack(tup)`.
- [ ] 96. Implement `split(ary, indices_or_sections, axis)`.
- [ ] 97. Implement `array_split(ary, indices_or_sections, axis)`.
- [ ] 98. Implement `tile(A, reps)`.
- [ ] 99. Implement `repeat(a, repeats, axis)`.
- [ ] 100. Implement `pad(array, pad_width, mode, constant_values)`.

### Phase 7: Logical & Relational Operations

- [ ] 101. Implement `equal(x1, x2)` / `x1.eq(x2)`.
- [ ] 102. Implement `not_equal(x1, x2)` / `x1.neq(x2)`.
- [ ] 103. Implement `less(x1, x2)` / `x1.lt(x2)`.
- [ ] 104. Implement `less_equal(x1, x2)` / `x1.lte(x2)`.
- [ ] 105. Implement `greater(x1, x2)` / `x1.gt(x2)`.
- [ ] 106. Implement `greater_equal(x1, x2)` / `x1.gte(x2)`.
- [ ] 107. Implement `logical_and(x1, x2)`.
- [ ] 108. Implement `logical_or(x1, x2)`.
- [ ] 109. Implement `logical_not(x)`.
- [ ] 110. Implement `logical_xor(x1, x2)`.
- [ ] 111. Implement `allclose(a, b, rtol, atol)`.
- [ ] 112. Implement `isclose(a, b, rtol, atol)`.
- [ ] 113. Implement `isnan(x)`.
- [ ] 114. Implement `isinf(x)`.
- [ ] 115. Implement `where(condition, x, y)`.

### Phase 8: Sorting, Searching, and Indexing

- [ ] 116. Implement `sort(a, axis)`.
- [ ] 117. Implement `argsort(a, axis)`.
- [ ] 118. Implement `nonzero(a)`.
- [ ] 119. Implement `extract(condition, arr)`.
- [ ] 120. Implement `take(a, indices, axis)` (mapping to ONNX `Gather`).
- [ ] 121. Implement `take_along_axis(arr, indices, axis)` (mapping to ONNX `GatherElements`).
- [ ] 122. Implement `put(a, ind, v, mode)`.
- [ ] 123. Implement `put_along_axis(arr, indices, values, axis)`.
- [ ] 124. Support basic slice syntax emulation in JS (`a.slice([start, stop, step])`).
- [ ] 125. Support multidimensional slicing `a.slice([ [start1, stop1], [start2, stop2] ])`.

### Phase 9: Advanced Neural Network Ops (onnx9000.nn)

- [ ] 126. Expose neural network ops seamlessly within the array API.
- [ ] 127. Implement `nn.relu(x)`.
- [ ] 128. Implement `nn.sigmoid(x)`.
- [ ] 129. Implement `nn.softmax(x, axis)`.
- [ ] 130. Implement `nn.log_softmax(x, axis)`.
- [ ] 131. Implement `nn.gelu(x)`.
- [ ] 132. Implement `nn.conv2d(x, w, b, strides, pads, dilations, groups)`.
- [ ] 133. Implement `nn.max_pool2d(x, kernel_shape, strides, pads)`.
- [ ] 134. Implement `nn.avg_pool2d(x, kernel_shape, strides, pads)`.
- [ ] 135. Implement `nn.batch_norm(x, scale, B, mean, var, epsilon)`.
- [ ] 136. Implement `nn.layer_norm(x, scale, B, axis, epsilon)`.
- [ ] 137. Implement `nn.dropout(x, ratio)`.
- [ ] 138. Implement `nn.linear(x, weight, bias)`.
- [ ] 139. Map all `nn` ops dynamically to their direct ONNX operator equivalents during Lazy building.
- [ ] 140. Expose standard loss functions natively (e.g., `nn.cross_entropy_loss`).

### Phase 10: Eager Execution Engine (WebGPU/WASM)

- [ ] 141. Ensure Eager mode immediately evaluates the ONNX AST for a single operation.
- [ ] 142. Compile tiny single-node ONNX graphs on the fly and execute via `onnx9000.runtime`.
- [ ] 143. Cache compiled micro-graphs (e.g., a simple `Add` graph) to prevent recompilation overhead on rapid looping.
- [ ] 144. Allow forced targeting for Eager mode (`np.set_device('webgpu')`).
- [ ] 145. Implement zero-copy buffer sharing between consecutive Eager ops running on WebGPU.
- [ ] 146. Track tensor reference counts to automatically free WebGPU memory for intermediate Eager tensors when no longer used.
- [ ] 147. Support explicit `.dispose()` calls on Tensors for tight memory loops.
- [ ] 148. Fallback to WASM Math automatically if a tensor is small enough (preventing GPU dispatch overhead).
- [ ] 149. Expose `.cpu()` and `.gpu()` methods on the Tensor object to force memory transfers.
- [ ] 150. Handle asynchronous execution naturally: math operations return Promises if WebGPU is active (`await a.add(b)`).

### Phase 11: Lazy Graph Builder (The Exporter)

- [ ] 151. Implement `np.lazy_mode(true)` to switch global context.
- [ ] 152. Implement `np.Input(name, shape, dtype)` to explicitly define graph ingress points.
- [ ] 153. When in Lazy mode, math operations return `LazyTensor` (representing an AST Edge) instead of data.
- [ ] 154. Implement AST node generation on math method calls (e.g., `a.add(b)` generates an ONNX `Add` node in the background).
- [ ] 155. Track topological order inherently as the user writes TS/JS code.
- [ ] 156. Implement `np.export_model(outputs, filename)` to serialize the AST to `.onnx`.
- [ ] 157. Ensure constants created in Lazy mode are correctly embedded into the `.onnx` as Initializers.
- [ ] 158. Auto-generate node names (`Add_1`, `MatMul_2`) if not explicitly provided by the user.
- [ ] 159. Support explicit naming: `a.add(b, { name: "MyAddition" })`.
- [ ] 160. Detect unused computational branches in Lazy mode and strip them automatically upon export.

### Phase 12: Graph Tracing & Python Integration

- [ ] 161. Implement a JS-equivalent to PyTorch `make_fx` / Tracing.
- [ ] 162. Allow passing a standard JS function `function myModel(x, y) { return x.add(y).mul(2); }` and auto-tracing it into an ONNX graph.
- [ ] 163. Handle native JS control flow (`if/else`) during tracing by either unwrapping dynamically or emitting ONNX `If` nodes via specialized hooks.
- [ ] 164. Export the TS API directly into Pyodide via JS-Py bindings.
- [ ] 165. Ensure Python code `z = np.add(x, y)` correctly calls the TS `onnx9000.array.add` under the hood.
- [ ] 166. Support Python operator overloading natively in Pyodide (`x + y` evaluates via the library).
- [ ] 167. Implement `__getitem__` and `__setitem__` in Python mapped to the slicing APIs.
- [ ] 168. Ensure output ONNX models are 100% compliant with standard Python `onnx-array-api` output.
- [ ] 169. Support exporting sub-graphs explicitly from traced functions.
- [ ] 170. Create decorators `@onnx_function` to enforce strict type checking before tracing.

### Phase 13: JIT Compilation (Lazy to WebGPU Shader Fusions)

- [ ] 171. If Lazy mode is active but the user calls `.numpy()` / `.evaluate()`, trigger a Just-In-Time compile.
- [ ] 172. Collapse chained elementwise operations (e.g., `(x * y) + z`) into a single, fused WGSL shader locally.
- [ ] 173. Execute the JIT-compiled macro-kernel on WebGPU instantly and return the result.
- [ ] 174. Discard the intermediate AST nodes once the macro-kernel is built (acting as an ultra-fast NumExpr equivalent for JS).
- [ ] 175. Cache JIT shaders based on the AST hash to speed up loop evaluations.
- [ ] 176. Provide explicit tuning APIs: `np.compile(myModel, { optimize: 'O3' })`.
- [ ] 177. If targeting WebNN, JIT compile the AST block directly to a WebNN `MLGraph` and execute.
- [ ] 178. Handle fallback gracefully: if an op cannot be fused, chunk the JIT block and execute standard micro-graphs.
- [ ] 179. Benchmark JIT fused execution vs naive Eager execution.
- [ ] 180. Provide verbose logging: `np.set_log_level('DEBUG')` to print generated WGSL shaders during JIT.

### Phase 14: NumPy Parity & Edge Cases

- [ ] 181. Ensure `NaN` and `Infinity` handling strictly matches NumPy IEEE-754 semantics.
- [ ] 182. Implement `np.nan_to_num(x)`.
- [ ] 183. Implement `np.clip(a, a_min, a_max)` matching exactly.
- [ ] 184. Implement `np.around(a, decimals)`.
- [ ] 185. Implement `np.fix(a)`.
- [ ] 186. Implement `np.i0(x)` (Modified Bessel function).
- [ ] 187. Implement `np.sinc(x)`.
- [ ] 188. Support `axis` parameter as tuples (e.g., `axis=(0, 2)`).
- [ ] 189. Resolve negative axes exactly as NumPy does (counting from the back).
- [ ] 190. Handle 0-D tensors (scalars) accurately, as ONNX handles them differently than older TF/NumPy versions.

### Phase 15: Quality Assurance & Testing

- [ ] 191. Write unit tests comparing JS `onnx9000.array` outputs natively against a running Python NumPy instance.
- [ ] 192. Ensure absolute tolerance (`atol`) and relative tolerance (`rtol`) limits are respected in test suites.
- [ ] 193. Create test suite verifying Lazy mode creates valid ONNX AST nodes for all 150+ math operations.
- [ ] 194. Execute the generated `.onnx` files through `onnxruntime-node` to ensure strict standard compliance.
- [ ] 195. Fuzz the JIT compiler with randomly chained elementwise operations.
- [ ] 196. Fuzz the shape broadcasting engine to ensure it mimics NumPy perfectly.
- [ ] 197. Validate memory leak absence in Eager WebGPU mode over 10,000 loop iterations.
- [ ] 198. Configure CI to run tests against Pyodide inside Headless Chrome.
- [ ] 199. Publish test coverage reports for the `onnx9000.array` module specifically.
- [ ] 200. Enforce strict TS typing, throwing compile-time errors if shapes/types mismatch in explicitly typed inputs.

### Phase 16: Interoperability with Ecosystem Tools

- [ ] 201. Support ingesting tensors generated by `onnx9000.transformers` feature extractors natively.
- [ ] 202. Allow using `onnx9000.array` within `onnx9000.modifier` custom JS node replacement scripts.
- [ ] 203. Integrate seamlessly with `@tensorflow/tfjs` tensors (providing bi-directional `.fromTfjs()` / `.toTfjs()` converters).
- [ ] 204. Integrate with Hugging Face `tokenizers` outputs natively.
- [ ] 205. Enable exporting generated ONNX models straight to `onnx9000.coreml` for iOS usage.
- [ ] 206. Export models straight to `onnx9000.iree` for AOT standalone JS execution.
- [ ] 207. Support ingesting standard JSON arrays from REST APIs transparently.
- [ ] 208. Implement `.toDataURL()` for rendering Image tensors (HWC, C=3/4) directly to HTML Canvas objects.
- [ ] 209. Implement `.toAudioBuffer()` for writing sequences directly to Web Audio API.
- [ ] 210. Implement standard CSV/TSV parsing natively into `Tensor` objects.

### Phase 17: String & Custom Data Types

- [ ] 211. Support ONNX `STRING` data types in the Eager array API.
- [ ] 212. Implement `np.char.add` (concatenating string tensors).
- [ ] 213. Implement `np.char.equal` (string matching).
- [ ] 214. Implement `np.char.replace`.
- [ ] 215. Implement Regex extract mapping to ONNX `RegexFullMatch` if opset allows.
- [ ] 216. Support custom complex numbers (`complex64`, `complex128`) by internally mapping to float arrays with trailing `[..., 2]` dimension.
- [ ] 217. Handle BFloat16 (`bfloat16`) casting natively.
- [ ] 218. Support quantized integer types natively (`uint8`, `int8`, `uint4`).
- [ ] 219. Expose dynamic quantization helpers directly on the Tensor: `a.quantize_dynamic()`.
- [ ] 220. Support boolean tensors cleanly, matching JS `true`/`false` mapping to Int8 `1`/`0`.

### Phase 18: Documentation & Developer Experience

- [ ] 221. Build comprehensive API docs mapping `numpy.X` to `onnx9000.array.X`.
- [ ] 222. Provide a "Rosetta Stone" mapping TF.js commands to `onnx9000` commands.
- [ ] 223. Include JSDoc comments directly on the methods to provide inline VSCode hovering.
- [ ] 224. Publish an interactive REPL on the documentation website to execute math live in the browser.
- [ ] 225. Provide tutorial: "Building a Custom Neural Network from Scratch in TypeScript using `onnx9000.array`".
- [ ] 226. Provide tutorial: "JIT Compiling WebGPU Shaders from JS Math".
- [ ] 227. Release as an independent NPM package `@onnx9000/array`.
- [ ] 228. Ensure tree-shaking works perfectly (importing `np.add` doesn't bundle the whole framework).
- [ ] 229. Write warning logs when developers trigger slow-paths (e.g., executing un-fusable ops forcing multiple GPU readbacks).
- [ ] 230. Build VSCode snippets for rapid model prototyping.

### Phase 19: Random Number Generation & Stateful APIs

- [ ] 231. Implement `np.random.rand()`.
- [ ] 232. Implement `np.random.randn()`.
- [ ] 233. Implement `np.random.randint()`.
- [ ] 234. Implement `np.random.uniform()`.
- [ ] 235. Implement `np.random.normal()`.
- [ ] 236. Implement `np.random.seed(seed)` to guarantee determinism across JS environments.
- [ ] 237. Ensure Random ops map to valid ONNX `RandomNormal` / `RandomUniform` nodes during Lazy execution.
- [ ] 238. Generate pseudo-random numbers efficiently via WASM algorithms (e.g., PCG or XorShift).
- [ ] 239. Handle ONNX stateful generation if required by specific opsets.
- [ ] 240. Implement stateful tracking for custom iteration loops built with the API.

### Phase 20: Final Polish and Release Readiness

- [ ] 241. Validate nested Tracing: A traced function calling another traced function emits a clean, flat ONNX graph.
- [ ] 242. Prevent name collisions globally when auto-generating node names for 10k+ nodes.
- [ ] 243. Allow inline ONNX graph optimization during export (`np.export_model(..., { optimize: 'O3' })`).
- [ ] 244. Implement `np.save` to serialize raw tensor data directly to `.npy` binary format.
- [ ] 245. Implement `np.load` to read `.npy` or `.npz` files directly from disk/URL.
- [ ] 246. Establish benchmarking suite measuring Eager dispatch overhead natively in V8 (Chrome).
- [ ] 247. Establish benchmarking suite measuring Eager dispatch overhead in JavaScriptCore (Firefox).
- [ ] 248. Provide clear "Not Implemented" exceptions for NumPy operations lacking any valid ONNX operator mapping.
- [ ] 249. Optimize GC pauses during heavy lazy graph construction strings.
- [ ] 250. Handle deeply nested tuples in `stack`/`concat` APIs.
- [ ] 251. Build in array indexing `a.get(0, 2, 1)` and `a.set(0, 2, 1, value)`.
- [ ] 252. Add a `np.vectorize` equivalent to map standard JS scalar math functions to ONNX parallel loops.
- [ ] 253. Map `np.meshgrid`.
- [ ] 254. Map `np.mgrid`.
- [ ] 255. Translate ONNX custom domains gracefully inside Eager mode.
- [ ] 256. Provide visual execution plan dumps using `console.table`.
- [ ] 257. Hook into standard `window.performance` API for fine-grained browser profiling.
- [ ] 258. Support memory profiling hooks `np.memory().gpu_allocated`.
- [ ] 259. Validate output ONNX models are immediately runnable in iOS CoreML bindings.
- [ ] 260. Implement `np.einsum_path` optimization utility.
- [ ] 261. Handle large tensor creation seamlessly by deferring to WebGPU Buffer mapping.
- [ ] 262. Check edge case zero-sized arrays.
- [ ] 263. Map string constants into proper UTF-8 encoded binary arrays.
- [ ] 264. Support generic sequence types in Python mapping to ONNX `SequenceProto`.
- [ ] 265. Enforce memory constraints on Pyodide environments implicitly.
- [ ] 266. Enable seamless JS <-> Pyodide memory sharing using `PyBuffer`.
- [ ] 267. Handle multi-threading in Eager mode using Web Workers explicitly.
- [ ] 268. Provide graceful WebGL fallbacks if WebGPU isn't available for Eager math.
- [ ] 269. Output a "Supported Numpy Ops" compatibility matrix file automatically during build.
- [ ] 270. Add support for creating sparse tensors directly via `np.sparse()`.
- [ ] 271. Implement specific matrix factorizations (`np.linalg.svd`) via CPU-bound WASM if GPU implementation is unstable.
- [ ] 272. Map specific `np.fft` routines to ONNX `DFT` (Discrete Fourier Transform) nodes if opset allows.
- [ ] 273. Support `np.pad` complex padding modes (e.g., `wrap`, `maximum`).
- [ ] 274. Implement advanced indexing using arrays of integers.
- [ ] 275. Handle bitwise operations natively (`bitwise_and`, `bitwise_or`).
- [ ] 276. Provide hooks for setting thread limits explicitly (`np.set_num_threads(4)`).
- [ ] 277. Validate parity with original `onnx-array-api` v0.2 Python implementations.
- [ ] 278. Add strict integration checking for ONNX models built exclusively in the browser and executed in C++.
- [ ] 279. Create custom exception classes (`BroadcastError`, `TypeMismatchError`) mirroring standard numerical libraries.
- [ ] 280. Establish automated NPM publish pipelines.
- [ ] 281. Enable users to override ONNX domain versions (`np.set_opset(18)`).
- [ ] 282. Add testing for extremely deep graph creation (tracing 1000+ operations in a loop).
- [ ] 283. Create custom memory leak detection for the JIT compilation engine.
- [ ] 284. Allow importing WebNN execution providers natively into the Eager dispatch loop.
- [ ] 285. Support executing Eager math using Apple Neural Engine natively on macOS Safari.
- [ ] 286. Handle ONNX node attributes dynamically updating based on Tensor sizes.
- [ ] 287. Expose raw WebGPU CommandEncoders for developers wishing to interleave their own graphics logic.
- [ ] 288. Manage memory layout conversion (NCHW vs NHWC) automatically depending on backend preference.
- [ ] 289. Track precise model memory bounds during trace evaluation to fail fast on Out Of Memory.
- [ ] 290. Finalize rigorous integration tests proving complete offline functionality.
- [ ] 291. Develop `np.polyfit` routines.
- [ ] 292. Support `np.histogram`.
- [ ] 293. Map `np.digitize`.
- [ ] 294. Enable custom tensor serialization formats.
- [ ] 295. Execute deep lifecycle analysis of Eager objects to prevent GC lockups.
- [ ] 296. Maintain strict exact parity against NumPy 1.26 functionality.
- [ ] 297. Support `--disable-webgpu-fp16` for legacy compatibility testing.
- [ ] 298. Validate precise execution under 1GB RAM bounds.
- [ ] 299. Write comprehensive API documentation for the `onnx9000.array` namespace.
- [ ] 300. Release v1.0 feature complete certification for `onnx9000.array`.
