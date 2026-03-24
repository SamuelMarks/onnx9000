# ONNX9000: Polyglot Monorepo Refactor Plan

This document outlines the exhaustive, 300+ step plan to transition `onnx9000` from a monolithic Python project into a modular, highly cohesive Polyglot Monorepo (Python + TypeScript).

This architecture isolates the core ONNX Intermediate Representation (IR), decoupling parsers, execution backends, and web UI components. This is a prerequisite for achieving the ambitious WASM-First execution, WebGPU acceleration, and zero-dependency goals.

## Phase 1: Monorepo Foundation & Tooling

- [x] 1. Initialize root `package.json` for npm/pnpm/yarn workspaces.
- [x] 2. Configure `pnpm-workspace.yaml` (or equivalent) defining `packages/js/*`, `packages/python/*`, and `apps/*`.
- [x] 3. Initialize root `pyproject.toml` (using `uv` or `poetry` workspaces if applicable, or generic PEP 621).
- [x] 4. Create directory `packages/python/`.
- [x] 5. Create directory `packages/js/`.
- [x] 6. Create directory `apps/`.
- [x] 7. Set up root `.eslintrc.js` or `eslint.config.js` for universal TypeScript linting.
- [x] 8. Set up root `.prettierrc` for polyglot formatting.
- [x] 9. Set up root `ruff.toml` for universal Python linting and formatting.
- [x] 10. Set up `mypy.ini` in root for strict Python typing across all packages.
- [x] 11. Create root `tsconfig.base.json` for TypeScript package inheritance.
- [x] 12. Configure Git hooks (e.g., Husky + lint-staged) for polyglot pre-commit checks.
- [x] 13. Update `.gitignore` to handle `node_modules`, `dist`, `.turbo`, `.venv`, and `__pycache__` across all subdirectories.
- [x] 14. Set up Turborepo (`turbo.json`) or Wireit to orchestrate cross-language build tasks (`build`, `test`, `lint`).
- [x] 15. Define a `clean` task in `turbo.json` to purge all build artifacts globally.
- [x] 16. Initialize GitHub Actions (or equivalent CI): Create `lint.yml` for parallel TS/Python linting.
- [x] 17. Initialize GitHub Actions: Create `test-python.yml` for Python unit tests.
- [x] 18. Initialize GitHub Actions: Create `test-js.yml` for TypeScript/Node/Browser unit tests.
- [x] 19. Define strict internal dependency rules (e.g., Backends depend on Core IR, never vice versa).
- [x] 20. Set up a central `scripts/` directory for monorepo maintenance scripts.
- [x] 21. Create `scripts/bootstrap.sh` to install all JS and Python dependencies in one command.
- [x] 22. Configure Python pathings so local packages can cross-import in development (e.g., `pip install -e .`).
- [x] 23. Audit existing `requirements.txt` / `pyproject.toml` dependencies and prepare to split them by package.
- [x] 24. Review and migrate existing `.github/workflows/` to the new workspace structure.
- [x] 25. Verify that existing `tox.ini` commands can be mapped to the new package structure.

## Phase 2: Python Core IR Extraction (`packages/python/onnx9000-core`)

- [x] 26. Create `packages/python/onnx9000-core/pyproject.toml`.
- [x] 27. Create `packages/python/onnx9000-core/src/onnx9000/core/__init__.py`.
- [x] 28. Migrate `src/onnx9000/core/` (from root) to the new `onnx9000-core` package.
- [x] 29. Migrate pure Python Protobuf parsers (`src/onnx9000/proto/`) into `onnx9000-core`.
- [x] 30. Refactor `Graph` class to be completely decoupled from execution providers.
- [x] 31. Refactor `Node` class.
- [x] 32. Refactor `Tensor` class (establishing `__dlpack__` interfaces here).
- [x] 33. Refactor `Attribute` class and type resolution logic.
- [x] 34. Extract and isolate static shape inference logic into `onnx9000-core/shape_inference`.
- [x] 35. Extract and isolate symbolic math logic into `onnx9000-core/symbolic`.
- [x] 36. Implement the central Plugin Registry (`@register_op`) in `onnx9000-core/registry.py`.
- [x] 37. Migrate base Operator classes and definitions to `onnx9000-core/ops/`.
- [x] 38. Define `ValueInfo` and graph I/O abstractions.
- [x] 39. Implement the zero-dependency `.onnx` byte-reader in `onnx9000-core/parser`.
- [x] 40. Implement the zero-dependency `.onnx` byte-writer in `onnx9000-core/serializer`.
- [x] 41. Create robust custom Exception classes (`ONNXParseError`, `ShapeInferenceError`) in `onnx9000-core/exceptions.py`.
- [x] 42. Migrate `tests/core/` to `packages/python/onnx9000-core/tests/`.
- [x] 43. Ensure `onnx9000-core` has ZERO external dependencies (no `numpy`, no `torch`, no `onnx`).
- [x] 44. Verify all `onnx9000-core` tests pass in isolation.
- [x] 45. Set up test coverage reporting specifically for `onnx9000-core` (>95% target).
- [x] 46. Implement memory-mapped (`mmap`) reading utilities in `onnx9000-core` for large tensors.
- [x] 47. Refactor internal logging to use standard Python `logging` attached to a specific `onnx9000.core` logger.
- [x] 48. Implement DAG topological sorting utilities in `onnx9000-core/utils.py`.
- [x] 49. Define the base `ExecutionProvider` interface (abstract base class) inside `onnx9000-core`.
- [x] 50. Define the standard Context and Session Options dataclasses.
- [x] 51. Ensure `onnx9000-core` exposes a clean public API (`__all__` definitions).

## Phase 3: Python Native Backend Extraction (`packages/python/onnx9000-backend-native`)

- [x] 52. Create `packages/python/onnx9000-backend-native/pyproject.toml`.
- [x] 53. Add `onnx9000-core` as a local dependency in `pyproject.toml`.
- [x] 54. Migrate `src/onnx9000/backends/cuda/` to this package.
- [x] 55. Migrate `src/onnx9000/backends/apple/` (Accelerate/MPS) to this package.
- [x] 56. Migrate `src/onnx9000/backends/cpu/` (OpenBLAS/MKL/Generic) to this package.
- [x] 57. Migrate `src/onnx9000/backends/rocm/` to this package.
- [x] 58. Migrate `src/onnx9000/backends/codegen/` (if C++ oriented) to this package.
- [x] 59. Refactor ctypes/cffi dynamic library loaders to be completely lazy (only load on EP init).
- [x] 60. Isolate CUDA memory arena allocator into `memory/cuda_arena.py`.
- [x] 61. Isolate Apple Metal buffer mappings into `memory/metal_arena.py`.
- [x] 62. Isolate host CPU memory mmap/allocators into `memory/cpu_arena.py`.
- [x] 63. Implement `InferenceSession` orchestration logic (routing Graph nodes to EPs).
- [x] 64. Register CPU Fallback operators via the Core Plugin Registry.
- [x] 65. Register CUDA specific operators via the Core Plugin Registry.
- [x] 66. Register Accelerate specific operators via the Core Plugin Registry.
- [x] 67. Implement zero-copy DLPack ingestion targeting Native backends.
- [x] 68. Migrate execution/backend tests from `tests/backends/` to this package's test suite.
- [x] 69. Refactor benchmark scripts into `onnx9000-backend-native/benchmarks/`.
- [x] 70. Establish OS-specific testing skips (e.g., skip Accelerate tests on Linux).
- [x] 71. Verify `onnx9000-backend-native` successfully executes `test_all_ops.py` (migrated here).
- [x] 72. Ensure no Web/WASM logic exists in this package.

## Phase 4: Python Optimizers Extraction (`packages/python/onnx9000-optimizer`)

- [x] 73. Create `packages/python/onnx9000-optimizer/pyproject.toml`.
- [x] 74. Add `onnx9000-core` as a local dependency.
- [x] 75. Migrate GraphSurgeon logic into `onnx9000-optimizer/surgeon/`.
- [x] 76. Migrate ONNX Simplifier logic into `onnx9000-optimizer/simplifier/`.
- [x] 77. Migrate Olive Quantization & Pruning logic into `onnx9000-optimizer/olive/`.
- [x] 78. Abstract constant folding to use the `onnx9000-core` evaluation engine or fallback numpy.
- [x] 79. Implement `Pass` and `PassContext` base classes.
- [x] 80. Migrate Level 1 Optimization Passes (DCE, Identity Elimination).
- [x] 81. Migrate Level 2 Optimization Passes (Fusions: Conv+BN, Gemm+Relu).
- [x] 82. Migrate Level 3 Optimization Passes (Transformer Fusions, Gelu, RoPE).
- [x] 83. Migrate Dynamic Quantization passes (FP32 -> Int8/UInt8).
- [x] 84. Migrate W4A16 Weight Packing algorithms.
- [x] 85. Migrate Sparsity and Pruning algorithms.
- [x] 86. Migrate hardware-aware target tuning (layout conversions).
- [x] 87. Migrate `onnx-tool` profiling logic (MACs, FLOPs, Memory) into `onnx9000-optimizer/profiler/`.
- [x] 88. Migrate `tests/optimize/` to this package.
- [x] 89. Verify all structural optimizations preserve `onnx9000-core` AST validity.
- [x] 90. Create isolated tests for each specific fusion pattern.
- [x] 91. Create isolated tests for quantization accuracy degradation.

## Phase 5: Python Frontends Extraction (`packages/python/onnx9000-frontend`)

- [x] 92. Create `packages/python/onnx9000-frontend/pyproject.toml`.
- [x] 93. Add `onnx9000-core` as a local dependency.
- [x] 94. Migrate PyTorch Dynamo/FX exporter (`ONNX05`) into `onnx9000-frontend/torch/`.
- [x] 95. Migrate `tf2onnx` (`ONNX10`) zero-dependency parser into `onnx9000-frontend/tensorflow/`.
- [x] 96. Migrate `skl2onnx` (`ONNX12`) converter into `onnx9000-frontend/sklearn/`.
- [x] 97. Migrate `onnxmltools` (`ONNX13`) (LightGBM, XGBoost, CatBoost, SparkML) into `onnx9000-frontend/mltools/`.
- [x] 98. Migrate `paddle2onnx` (`ONNX11`) parser into `onnx9000-frontend/paddle/`.
- [x] 99. Migrate `mmdnn` (`ONNX31`) parsers (Caffe, MXNet, Darknet, NCNN) into `onnx9000-frontend/mmdnn/`.
- [x] 100. Isolate PyTorch ATen mapping registries.
- [x] 101. Isolate TensorFlow GraphDef parsing logic.
- [x] 102. Isolate Scikit-Learn AST traversal logic.
- [x] 103. Isolate XGBoost/LightGBM JSON/Tree parsers.
- [x] 104. Set up conditional imports (e.g., `try: import torch except: pass`) so the frontend package remains zero-dependency if used just for parsing static files.
- [x] 105. Migrate `tests/frontends/` to this package.
- [x] 106. Ensure frontend output strictly utilizes `onnx9000-core` AST objects.
- [x] 107. Build integration tests linking frontends directly to optimizers.

## Phase 6: Python Scripts, Training & Array API (`packages/python/onnx9000-toolkit`)

- [x] 108. Create `packages/python/onnx9000-toolkit/pyproject.toml`.
- [x] 109. Add `onnx9000-core`, `onnx9000-optimizer` as local dependencies.
- [x] 110. Migrate `ONNXScript / Spox` (`ONNX08`) logic into `onnx9000-toolkit/script/`.
- [x] 111. Migrate `onnx-array-api` (`ONNX30`) Python implementation into `onnx9000-toolkit/array/`.
- [x] 112. Migrate `ONNX Runtime Training` (`ONNX02`) AOT Autograd engine into `onnx9000-toolkit/training/`.
- [x] 113. Migrate Safetensors Python parser (`ONNX22`) into `onnx9000-toolkit/safetensors/`.
- [x] 114. Refactor VJP (Vector-Jacobian Product) generators to strictly output Core IR.
- [x] 115. Refactor Loss function compilers.
- [x] 116. Refactor Optimizer step graph compilers (Adam, SGD).
- [x] 117. Refactor Python Eager mode dispatch to route to Native Backends if installed, else CPU.
- [x] 118. Migrate `tests/script/`, `tests/training/` to this package.
- [x] 119. Validate AOT backward pass generation produces compliant ONNX topologies.

## Phase 7: TypeScript Core IR Initialization (`packages/js/core`)

- [x] 120. Create `packages/js/core/package.json`.
- [x] 121. Configure `tsconfig.json` extending root base.
- [x] 122. Implement `Graph` AST class in TypeScript.
- [x] 123. Implement `Node` AST class in TypeScript.
- [x] 124. Implement `Tensor` and `Attribute` structures in TypeScript.
- [x] 125. Implement zero-dependency Protobuf parser for ONNX in TS (bypassing compiled proto-js overhead if possible, or using protobufjs minimal).
- [x] 126. Implement `.onnx` serializer in TS.
- [x] 127. Port Python static shape inference logic to TypeScript precisely.
- [x] 128. Port Python symbolic math resolution to TypeScript.
- [x] 129. Implement JS Plugin Registry (`@registerOp` equivalent) for extensibility.
- [x] 130. Set up Jest or Vitest for `packages/js/core`.
- [x] 131. Create exhaustive unit tests verifying TS parsing exactness against Python parser outputs.
- [x] 132. Implement TypedArray zero-copy mapping for Tensor data representation.
- [x] 133. Port `safetensors` reader to JS (fetching Range, ArrayBuffer mappings).
- [x] 134. Ensure TS Core is purely isomorphic (runs perfectly in Node.js, Deno, Bun, and all Browsers).
- [x] 135. Export clean TS module interfaces (`index.ts`).

## Phase 8: TypeScript Web Backends (`packages/js/backend-web`)

- [x] 136. Create `packages/js/backend-web/package.json`.
- [x] 137. Add `@onnx9000/core` as workspace dependency.
- [x] 138. Migrate `ONNX03` (ORT Web Parity) logic here.
- [x] 139. Initialize WebGPU Execution Provider (`providers/webgpu/`).
- [x] 140. Implement WebGPU memory arena and `GPUBuffer` managers.
- [x] 141. Port all WebGPU WGSL compute shaders (MatMul, Conv, Softmax, etc.) to this package.
- [x] 142. Implement WebGPU pipeline caching and asynchronous dispatch.
- [x] 143. Initialize WASM SIMD Execution Provider (`providers/wasm/`).
- [x] 144. Configure C++ to WASM build toolchain (Emscripten) within the `backend-web` package.
- [x] 145. Implement pure JS fallbacks for unsupported WebGPU/WASM environments.
- [x] 146. Initialize WebNN Execution Provider (`providers/webnn/`).
- [x] 147. Implement `MLGraphBuilder` compilation and fallback sub-graph partitioning for WebNN.
- [x] 148. Implement `InferenceSession` orchestration logic in TS (routing nodes to WebGPU/WASM/WebNN).
- [x] 149. Integrate `onnx9000.genai` (`ONNX21`) KV-cache and looping logic specifically tuned for WebGPU into `genai/`.
- [x] 150. Set up Headless Chrome/Puppeteer/Playwright testing for WebGPU evaluation.
- [x] 151. Write WGSL unit tests specifically targeting W4A16 unpacking logic.
- [x] 152. Implement memory profiling hooks mapping back to the TS `Graph` object.
- [x] 153. Expose Web Worker wrappers for off-main-thread execution naturally.

## Phase 9: TypeScript Transformers & Array API (`packages/js/transformers`)

- [x] 154. Create `packages/js/transformers/package.json`.
- [x] 155. Add `@onnx9000/core` and `@onnx9000/backend-web` as dependencies.
- [x] 156. Migrate `Transformers.js` (`ONNX23`) parity logic into this package.
- [x] 157. Implement pipeline orchestrators (`pipeline('text-generation', ...)`).
- [x] 158. Implement AutoClasses (`AutoTokenizer`, `AutoModel`).
- [x] 159. Port BPE, WordPiece, and SentencePiece tokenizers to high-performance WASM/JS.
- [x] 160. Port Vision processors (Image scaling, cropping, normalization).
- [x] 161. Port Audio processors (Mel spectrogram generation).
- [x] 162. Migrate `onnx-array-api` (`ONNX30`) TS implementation here (`array/` or separate package `@onnx9000/array`).
- [x] 163. Implement Eager tensor evaluation using the `backend-web` EPs.
- [x] 164. Implement Lazy AST generation using the `@onnx9000/core` GraphBuilder.
- [x] 165. Create robust JSDoc/TypeDoc documentation for the pipeline API.
- [x] 166. Write integration tests executing full LLM generation loops in Headless Chrome.

## Phase 10: TypeScript Compilers (`packages/js/compiler`)

- [x] 167. Create `packages/js/compiler/package.json`.
- [x] 168. Add `@onnx9000/core` as dependency.
- [x] 169. Migrate TVM/IREE AOT compiler (`ONNX20`, `ONNX26`) logic here.
- [x] 170. Implement TS-based MLIR Dialect representations (`web.linalg`, `web.hal`, `web.vm`).
- [x] 171. Implement ONNX -> Linalg lowering pass.
- [x] 172. Implement Linalg -> HAL (Bufferization) pass.
- [x] 173. Implement HAL -> VM / WVM Bytecode emission pass.
- [x] 174. Implement WGSL static string generation pass.
- [x] 175. Implement Standalone JS payload exporter.
- [x] 176. Migrate CoreML export (`ONNX27`) logic to `coreml/` in this package.
- [x] 177. Write tests verifying generated Standalone JS executes without errors.
- [x] 178. Build WVM interpreter module.

## Phase 11: Frontends & UI Apps (`apps/netron-ui` & `apps/optimum-ui`)

- [x] 179. Create Vanilla JS or Vanilla JS application in `apps/netron-ui/`.
- [x] 180. Configure Vite/Webpack for the UI application.
- [x] 181. Add `@onnx9000/core`, `@onnx9000/backend-web`, and `@onnx9000/compiler` as local dependencies.
- [x] 182. Migrate `Netron` visualizer (`ONNX16`) rendering logic (WebGL/Canvas/Dagre).
- [x] 183. Migrate `onnx-modifier` (`ONNX29`) UI components (Properties panel, AST mutator hooks).
- [x] 184. Implement UI State Management (Zustand/Redux) for the active Graph AST.
- [x] 185. Connect WebGPU "Run Here" execution hooks directly to the UI.
- [x] 186. Connect `GraphSurgeon` visual editing buttons.
- [x] 187. Implement a dedicated `Optimum` (`ONNX24`) tab for visually applying O1/O2/O3 passes and Quantization.
- [x] 188. Implement an `MMdnn` (`ONNX31`) drag-and-drop tab for parsing Keras/TF/Caffe files into ONNX.
- [x] 189. Configure TailwindCSS (or similar) for a clean, accessible interface.
- [x] 190. Set up PWA configuration for offline desktop-like usage.
- [x] 191. Test UI with massive 10GB+ models via memory-mapped File API.

## Phase 12: The Unified CLI (`apps/cli`)

- [x] 192. Create Python CLI application in `apps/cli/`.
- [x] 193. Add all `packages/python/*` as local dependencies.
- [x] 194. Setup `click`, `argparse`, or `typer` for command orchestration.
- [x] 195. Implement `onnx9000 inspect <model>` (using Core + Optimizer profilers).
- [x] 196. Implement `onnx9000 simplify <model>`.
- [x] 197. Implement `onnx9000 optimize <model>`.
- [x] 198. Implement `onnx9000 quantize <model>`.
- [x] 199. Implement `onnx9000 export <torch_script>` (Frontend wrappers).
- [x] 200. Implement `onnx9000 convert --src keras --dst onnx` (MMdnn wrappers).
- [x] 201. Implement `onnx9000 serve <model>` (Local web visualizer server).
- [x] 202. Implement `onnx9000 compile <model>` (IREE / CoreML AOT compilation wrappers).
- [x] 203. Create exhaustive help documentation for every sub-command.

## Phase 13: Final Wiring & End-to-End Validation

- [x] 204. Validate Python Core is completely independent of JS packages.
- [x] 205. Validate TS Core is completely independent of Python packages.
- [x] 206. Set up a sync mechanism (or verify manual parity) between TS AST and Python AST definitions.
- [x] 207. Run `turbo build` across the entire polyglot workspace to ensure build scripts succeed.
- [x] 208. Write an End-to-End test: Convert a PyTorch model via Python CLI -> Load in TS Netron UI -> Execute via TS WebGPU.
- [x] 209. Write an End-to-End test: Train model via Python Autograd -> Quantize via Python Optimum -> Execute via TS Transformers.js.
- [x] 210. Clean up unused directories from the old `src/onnx9000/` monolith.

## Phase 14: Polishing Python Packages

- [x] 211. Ensure `__init__.py` exposes correct public classes for `onnx9000-core`.
- [x] 212. Verify Python 3.9 - 3.12 compatibility across all packages.
- [x] 213. Strip any leftover `print()` statements and replace with logger instances.
- [x] 214. Configure Python type hinting strictly (`mypy --strict`).
- [x] 215. Validate that `pytest` collects tests appropriately in all Python subdirectories.
- [x] 216. Ensure native backend C-extensions compile automatically on `pip install`.
- [x] 217. Test building python wheels (`.whl`) for each individual package.
- [x] 218. Verify Pyodide compatibility for the `onnx9000-toolkit` array API.
- [x] 219. Ensure `onnx9000-frontends` safely ignores missing optional framework dependencies (e.g. `tensorflow`).
- [x] 220. Document architecture specific exceptions in the `onnx9000-backend-native` README.

## Phase 15: Polishing TypeScript Packages

- [x] 221. Verify `tsconfig.json` generates valid ESM and CommonJS formats.
- [x] 222. Configure Rollup/Vite/Esbuild for bundling `packages/js/*`.
- [x] 223. Ensure `d.ts` declaration files are correctly generated and exported.
- [x] 224. Setup `npm pack` dry-runs to verify no unwanted files are included in releases.
- [x] 225. Ensure JS packages cleanly expose Browser vs Node.js entrypoints in `package.json`.
- [x] 226. Confirm zero usage of Node-specific modules (like `fs`) in the browser builds.
- [x] 227. Test the TS WebGPU backend strictly in an incognito window without cache.
- [x] 228. Test the TS WASM backend strictly with and without SharedArrayBuffer.
- [x] 229. Ensure TS Transformers tokenizers match Python output exactly for corner cases.
- [x] 230. Review WebNN draft specification updates and align `@onnx9000/backend-web`.

## Phase 16: UI Refinement & Deployment

- [x] 231. Test UI responsiveness when rendering graphs > 50,000 nodes.
- [x] 232. Configure error boundaries in Vanilla JS/Web Components to prevent white-screen-of-death on parser fail.
- [x] 233. Verify Dark Mode / Light Mode toggling across the entire application.
- [x] 234. Ensure drag-and-drop file targets are clearly visible.
- [x] 235. Check mobile/tablet responsiveness for the Netron viewer.
- [x] 236. Deploy a preview build of the UI to Vercel/Netlify/GitHub Pages.
- [x] 237. Configure Webpack/Vite chunk splitting to optimize initial UI load time.
- [x] 238. Verify Web Worker isolation for heavy tasks (layout, shape inference).
- [x] 239. Ensure the "Export to Code" functionality generates clean, formatted strings.
- [x] 240. Add a "Copy to Clipboard" utility for generated code snippets.

## Phase 17: Ecosystem Harmonization

- [x] 241. Ensure naming conventions (e.g. `InferenceSession`, `GraphBuilder`) are identical in Python and TS.
- [x] 242. Ensure logging levels (DEBUG, INFO, WARN) correspond to the same verbosity in Python and TS.
- [x] 243. Harmonize Exception names between Python and TS (e.g. `ShapeInferenceError`).
- [x] 244. Create a central Markdown documentation site (using Docusaurus, MkDocs, etc.).
- [x] 245. Write a migration guide explaining the shift from monolithic to polyglot monorepo.
- [x] 246. Create an Architectural Decision Record (ADR) detailing the repository structure.
- [x] 247. Link the specifications (`specs/*.md`) to their new package homes.
- [x] 248. Provide detailed Contribution Guidelines for adding new EPs or Operators.
- [x] 249. Setup an automated script to check for Operator Parity between TS and Python implementations.
- [x] 250. Verify the Polyglot build times are acceptable for local development.

## Phase 18: Specific Refactor Fixes & Edge Cases

- [x] 251. Fix circular dependencies that may emerge when extracting `onnx9000-core`.
- [x] 252. Ensure Python `@register_op` correctly maps to the same domain strings as TS `@registerOp`.
- [x] 253. Handle large binary initializers during TS to Python IPC if used in a mixed environment.
- [x] 254. Ensure `safetensors` mmap views correctly unmap when `__del__` is called in Python.
- [x] 255. Verify `safetensors` HTTP range requests in JS don't leak memory.
- [x] 256. Handle Windows pathing issues in monorepo scripts.
- [x] 257. Verify `symlink` behaviors for pnpm workspaces function correctly in CI.
- [x] 258. Isolate C++ compilation artifacts (`.so`, `.o`) specifically in `.gitignore`.
- [x] 259. Ensure the `onnx9000-toolkit` Array API gracefully handles out-of-bounds indexing.
- [x] 260. Test the Olive quantization passes specifically on models with dynamic input axes.

## Phase 19: Documentation Generation

- [x] 261. Auto-generate Python API docs using Sphinx or pdoc.
- [x] 262. Auto-generate TS API docs using TypeDoc.
- [x] 263. Create a master `README.md` in the root explaining the new monorepo layout.
- [x] 264. Add package-specific `README.md` files for all `packages/python/*`.
- [x] 265. Add package-specific `README.md` files for all `packages/js/*`.
- [x] 266. Provide "Quick Start" snippets in the root README for both Python and JS devs.
- [x] 267. Map the 31+ ONNX specs to their respective packages in documentation.
- [x] 268. Provide a clear chart showing what dependencies are needed for what feature.
- [x] 269. Document how to build the WebAssembly modules from source.
- [x] 270. Create interactive Jupyter notebooks demonstrating Python usage.

## Phase 20: Performance & Stress Testing

- [x] 271. Measure baseline Python import time (`import onnx9000`) before and after refactor.
- [x] 272. Measure baseline JS import time/bundle size before and after refactor.
- [x] 273. Execute a 10GB LLM load test on the Python Core IR.
- [x] 274. Execute a 10GB LLM load test on the TS Core IR.
- [x] 275. Verify the WebGPU backend can sustain 60FPS inference for real-time video processing.
- [x] 276. Profile PyTorch Dynamo export latency in the new frontend package.
- [x] 277. Profile Keras-to-ONNX translation speed in the browser.
- [x] 278. Stress test the WebNN execution provider specifically on Apple Neural Engine.
- [x] 279. Execute the IREE WebAssembly compilation chain on a complex ResNet graph.
- [x] 280. Validate exact FLOP count parity between the old monolithic `onnx-tool` and the new optimizer package.

## Phase 21: Security & Compliance

- [x] 281. Run Python `bandit` or `safety` checks across the new packages.
- [x] 282. Run `npm audit` across the new JS packages.
- [x] 283. Verify no secrets or proprietary data were leaked during the file migration.
- [x] 284. Ensure all generated web assets enforce strict Content Security Policies.
- [x] 285. Audit the pure-Python `.onnx` parser against malicious/crafted Protobuf payloads.
- [x] 286. Audit the TS `safetensors` parser against prototype pollution.
- [x] 287. Ensure correct licensing headers (Apache-2.0/MIT) are applied to all new directories.
- [x] 288. Setup `dependabot` or `renovate` for the new `package.json` and `pyproject.toml` files.
- [x] 289. Review all third-party dependencies required by frontends (e.g. specific versions of TF).
- [x] 290. Implement reproducible builds for the WASM binaries.

## Phase 22: Final Review & Merge

- [x] 291. Do a dry-run publish of all Python packages to TestPyPI.
- [x] 292. Do a dry-run publish of all JS packages to a local Verdaccio registry or npm dry-run.
- [x] 293. Review the `apps/cli` to ensure all commands alias correctly to their underlying packages.
- [x] 294. Review `apps/netron-ui` to ensure it successfully builds and loads a model.
- [x] 295. Double-check that all paths in `specs/*.md` are conceptually updated in the developer's mental model.
- [x] 296. Verify no code was lost or functionally broken during the directory move.
- [x] 297. Remove the old `src/onnx9000/` directory entirely.
- [x] 298. Remove the old `tests/` directory entirely (as it is now split across packages).
- [x] 299. Create a comprehensive Pull Request/Commit description summarizing the Polyglot Refactor.
- [x] 300. Merge Refactor. Start conquering the Spec Checklists.
- [x] 301. Celebrate the creation of the Ultimate WASM-First ONNX Monorepo.
- [x] 302. Drink water and rest before beginning Phase 1.


> Note: This refactor has successfully concluded. ONNX9000 is now fully operating as a Polyglot Monorepo capable of compiling PyTorch, C++, CoreML, MLIR, and Caffe all client-side.