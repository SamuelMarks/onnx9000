# onnx-safetensors Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `safetensors` within the `onnx9000` ecosystem.
The original `safetensors` library (by Hugging Face) relies on Rust bindings to provide secure, fast, and zero-copy tensor serialization.
Our `onnx9000` reimplementation provides a 100% pure-Python and pure-JavaScript (WASM/WebGPU) parser. It reads `.safetensors` files directly, mapping them securely to ONNX `TensorProto` data structures. By leveraging native POSIX `mmap` in Python and `ArrayBuffer` / `fetch` Range requests in the browser, we can progressively stream and instantly load multi-gigabyte LLM weights into memory-constrained environments with true zero-copy performance and absolute security (no `pickle` vulnerabilities).

## Exhaustive Parity Checklist

### 1. Pure-Python/JS Core Parsing Engine (40+ items)

- [ ] Implement zero-dependency `.safetensors` header parser in Python
- [ ] Implement zero-dependency `.safetensors` header parser in JavaScript (TypeScript)
- [ ] Read 8-byte little-endian unsigned integer (header size `N`)
- [ ] Read standard JSON header payload of exact length `N`
- [ ] Validate JSON encoding strictly as UTF-8
- [ ] Extract tensor metadata: `dtype`
- [ ] Extract tensor metadata: `shape`
- [ ] Extract tensor metadata: `data_offsets` `[begin, end]`
- [ ] Implement O(1) dictionary lookups for tensor names
- [ ] Verify `begin` and `end` offsets fit within the bounds of the file size
- [ ] Verify `end - begin` exactly matches the calculated byte size of `shape` \* `dtype_size`
- [ ] Support implicit 8-byte alignment padding validation
- [ ] Ignore internal JSON whitespace formatting variants seamlessly
- [ ] Reject duplicate tensor names in the JSON header
- [ ] Reject overlapping tensor data regions within the file buffer
- [ ] Reject tensor data regions that precede the end of the JSON header
- [ ] Extract global `__metadata__` dictionary natively
- [ ] Provide lazy-evaluation generator yielding tensor slices on demand
- [ ] Close file handles gracefully upon parser garbage collection
- [ ] Expose API to list all tensors without reading binary payload (`.keys()`)
- [ ] Support parsing strictly from an in-memory byte buffer (e.g., downloaded RAM)
- [ ] Support parsing from a generic `io.BytesIO` / `io.BufferedReader` stream
- [ ] Provide strict boundary error messages for corrupted headers
- [ ] Ensure nested or invalid JSON types in `__metadata__` do not crash the parser
- [ ] Implement fast path for extracting single tensors by name
- [ ] Implement bulk extraction for multiple tensors
- [ ] Support indexing into tensor lists natively `tensors['weight']`
- [ ] Expose total byte footprint calculator for memory planning
- [ ] Emulate `safetensors.safe_open` API signature for PyTorch ecosystem compatibility
- [ ] Enable framework-agnostic tensor abstraction (`onnx9000.Tensor` object mapping)
- [ ] Provide an explicit validation-only mode (`check_safetensors(file)`)
- [ ] Reject files exceeding JS `Number.MAX_SAFE_INTEGER` boundaries for offsets safely
- [ ] Parse Hugging Face sharded formats (`model-00001-of-00002.safetensors`)
- [ ] Aggregate sharded JSON indexes (`model.safetensors.index.json`) logically
- [ ] Implement global dictionary view spanning multiple sharded `.safetensors` files
- [ ] Detect and warn about unreferenced data (bytes in the file not mapped to any tensor)
- [ ] Reject unaligned tensor offsets (must be 8-byte aligned per standard)

### 2. Zero-Copy Memory & Mmap Implementation (30+ items)

- [ ] Implement POSIX `mmap` for instant disk-to-memory mapping natively in Python
- [ ] Implement Windows `mmap` equivalent (`CreateFileMapping`) safely
- [ ] Extract NumPy `ndarray` views directly from the `mmap` buffer (zero-copy)
- [ ] Ensure NumPy arrays generated are marked as read-only (`writeable=False`)
- [ ] Prevent Python Garbage Collector from closing `mmap` while views are alive
- [ ] Support explicit `madvise` hints (`MADV_WILLNEED`, `MADV_RANDOM`) for OS page caching
- [ ] Implement Pyodide `FS` virtual filesystem zero-copy extraction
- [ ] Map ArrayBuffers in WebAssembly explicitly from the Safetensors buffer
- [ ] Support zero-copy injection of weights directly into WebGPU `GPUBuffer`
- [ ] Implement `memoryview` based slicing for environments without NumPy
- [ ] Avoid intermediate byte string allocations (`file.read()`) during standard extraction
- [ ] Guarantee memory layout contiguity (C-contiguous) for all extracted tensors
- [ ] Extract multi-dimensional slices lazily (e.g., `tensor[0:10, :]` via numpy strides)
- [ ] Handle explicit file-descriptor leaking protections (Context Managers)
- [ ] Support `mmap` across multiple processes via OS shared memory semantics
- [ ] Provide memory-pinned buffer extraction (CUDA Pinned Memory emulation) if requested
- [ ] Verify page alignment optimizations natively
- [ ] Prevent swapping to disk dynamically on specific critical tensors via `mlock`
- [ ] Handle multi-gigabyte models on 32-bit WASM gracefully (falling back to chunked arrays)
- [ ] Emulate 64-bit memory addressing for WASM-64 future compatibility
- [ ] Share Safetensors `mmap` memory maps across Python threads cleanly

### 3. Web-Specific Progressive Streaming & HTTP (40+ items)

- [ ] Implement HTTP `Range` request wrapper for native JS `fetch`
- [ ] Download ONLY the JSON header using initial 8-byte + N-byte HTTP requests
- [ ] Stream specific tensor payloads dynamically using `Range: bytes=begin-end`
- [ ] Assemble distributed LLM weights exclusively on demand (e.g., layer by layer loading)
- [ ] Integrate with WebGPU chunked pipeline execution (stream layer, execute, discard layer)
- [ ] Support HTTP `Keep-Alive` explicitly for thousands of rapid range requests
- [ ] Implement ServiceWorker caching for downloaded tensor chunks (`CacheStorage`)
- [ ] Implement IndexedDB persistence for entire `.safetensors` files in the browser
- [ ] Provide visual progress bar hooks for `Content-Length` streams
- [ ] Parse `Accept-Ranges: bytes` headers to fallback gracefully if streaming is blocked
- [ ] Support WebSockets / WebRTC for P2P tensor weight distribution in browser
- [ ] Stream `.safetensors` directly from Hugging Face Hub URLs seamlessly
- [ ] Implement auto-retry logic with exponential backoff for interrupted Range requests
- [ ] Decrypt HTTP chunked encodings securely
- [ ] Expose native `ReadableStream` interfaces for streaming JSON parsers
- [ ] Prevent browser Out-Of-Memory by actively destroying `Uint8Array` views after WGSL upload
- [ ] Manage parallel chunk downloads (e.g., fetching 4 layers simultaneously)
- [ ] Throttle parallel requests to prevent browser network stack blocking
- [ ] Support generating standard `ai.onnx` models entirely client-side using downloaded weights
- [ ] Enable cross-origin resource sharing (CORS) header pre-flight validations explicitly
- [ ] Handle 416 Range Not Satisfiable errors gracefully
- [ ] Fallback to full file download if server ignores Range headers

### 4. Data Type, Endianness & Tensor Alignment (40+ items)

- [ ] Parse `F64` -> `Float64` / `float64`
- [ ] Parse `F32` -> `Float32` / `float32`
- [ ] Parse `F16` -> `Float16` / `float16`
- [ ] Parse `BF16` -> `BFloat16` / `bfloat16`
- [ ] Parse `I64` -> `Int64` / `int64`
- [ ] Parse `I32` -> `Int32` / `int32`
- [ ] Parse `I16` -> `Int16` / `int16`
- [ ] Parse `I8` -> `Int8` / `int8`
- [ ] Parse `U64` -> `UInt64` / `uint64`
- [ ] Parse `U32` -> `UInt32` / `uint32`
- [ ] Parse `U16` -> `UInt16` / `uint16`
- [ ] Parse `U8` -> `UInt8` / `uint8`
- [ ] Parse `BOOL` -> `Bool` / `bool`
- [ ] Implement fallback byte-swapping if host architecture is Big-Endian
- [ ] Ensure strict Little-Endian decoding for all types natively
- [ ] Parse complex types (`C64`, `C128`) if standard evolves, or map to F32/F64 pairs
- [ ] Reject unrecognized or proprietary data types securely
- [ ] Emulate `bfloat16` natively in standard JavaScript (`Float32Array` masking)
- [ ] Validate standard HuggingFace specific type strings (e.g., `F16` vs `FLOAT16`)
- [ ] Implement explicit downcasting hooks (e.g., load `F32` but immediately return `F16` view)
- [ ] Implement explicit quantization hooks (load `F16`, convert to INT8 array dynamically)
- [ ] Extract tensor dimensionality explicitly (1D, 2D, 3D, 4D, ND)
- [ ] Throw error on 0-dimensional scalars if not correctly encoded as `shape: []`
- [ ] Handle massive dimensions safely (`shape: [1, 32, 128000, 128]`)
- [ ] Verify `int64` bounds securely (preventing Python `OverflowError` during slicing)
- [ ] Decode sub-byte quantization (e.g., NF4, INT4) explicitly via byte unpacking strategies
- [ ] Unpack specific AWQ / GPTQ packed `safetensors` layouts correctly

### 5. ONNX Graph Integration & Surgery (30+ items)

- [ ] Convert `.safetensors` mappings directly into ONNX `Initializer` tensors
- [ ] Replace standard `.bin` external data natively with `.safetensors` lookups
- [ ] Intercept ONNX parsing to pull constants exclusively from `.safetensors` indices
- [ ] Strip raw byte arrays from `ModelProto` and dump to `.safetensors` dynamically
- [ ] Export ONNX model to a `.onnx` (topology only) and `.safetensors` (weights only) pair
- [ ] Inject `GraphSurgeon` parameters directly from loaded safetensors dictionaries
- [ ] Rewrite ONNX `Constant` nodes seamlessly into `safetensors` memory views
- [ ] Support initializing an `onnx9000` Python compiled model from a `.safetensors` file
- [ ] Validate ONNX topological shapes against `safetensors` extracted shapes at runtime
- [ ] Warn on shape mismatches between ONNX ValueInfo and Safetensor arrays
- [ ] Warn on dtype mismatches between ONNX ValueInfo and Safetensor arrays
- [ ] Emulate standard ONNX Runtime `SessionOptions` external data configurations
- [ ] Pack multi-layer ONNX transformers heavily using shared `.safetensors` allocations
- [ ] Flatten nested Safetensors attributes into Graph inputs securely
- [ ] Validate `ai.onnx` operators evaluate correctly against the extracted views

### 6. Security, Auditing & Validation (The "Safe" in Safetensors) (30+ items)

- [ ] Prevent Arbitrary Code Execution (0-day vulnerabilities vs Python `pickle`)
- [ ] Enforce strict sandboxing of file parsing logic
- [ ] Validate `shape` arrays contain no negative dimensions
- [ ] Validate `shape` arrays contain no impossibly large dimensions (e.g., > 2^50)
- [ ] Prevent Path Traversal vulnerabilities in sharded `index.json` path loading
- [ ] Prevent XML External Entity (XXE) or JSON injection in header parsing
- [ ] Reject `.safetensors` files where `end` offset is smaller than `begin` offset
- [ ] Ensure byte offsets strictly respect the total file size reported by the OS
- [ ] Catch dynamic schema mutation attacks (JSON tampering)
- [ ] Validate `__metadata__` strings do not contain executable script tags (XSS protection for browsers)
- [ ] Fuzz-test parser against heavily corrupted JSON headers
- [ ] Fuzz-test parser against heavily corrupted binary offsets
- [ ] Provide cryptographic hashing verification (`SHA256`) of the file buffer optionally
- [ ] Verify metadata signatures (if standardized securely by HuggingFace)
- [ ] Ensure parser runs securely within Cloudflare Workers isolates

### 7. Distributed Server & High-Performance IO (30+ items)

- [ ] Deploy Safetensors `mmap` views natively in Ray Clusters for zero-copy IPC
- [ ] Serialize `onnx9000.Tensor` wrappers across gRPC efficiently using Safetensors formats
- [ ] Implement lazy-loading for Celery distributed background workers
- [ ] Use `sendfile` or `splice` Linux syscalls explicitly for networking weights from disk
- [ ] Optimize AWS S3 `boto3` integration with HTTP Range requests natively
- [ ] Optimize Azure Blob Storage `get_blob_client` with Range offsets
- [ ] Optimize GCP Cloud Storage chunked loading directly into memory
- [ ] Maximize Page Cache utilization on NVMe arrays (reading 70B parameters < 2 seconds)
- [ ] Provide distributed sharding algorithms natively (e.g., loading only layer 1-10 on Node A)
- [ ] Expose native `MPI` rank loading filters (Node $i$ only loads Safetensor array $i$)
- [ ] Pipeline parallelism loading strategies (Stream Layer N+1 while computing Layer N)
- [ ] Tensor parallelism loading strategies (Load slice `[:, 0:Dim/2]` directly from disk)

### 8. Serialization, Exporting & Creation (40+ items)

- [ ] Implement `.safetensors` writing logic purely in Python
- [ ] Implement `.safetensors` writing logic purely in Javascript (Node.js/Browser)
- [ ] Accept generic Python dictionaries (`dict[str, np.ndarray]`) for serialization
- [ ] Accept ONNX `TensorProto` arrays for serialization
- [ ] Generate strict UTF-8 JSON headers
- [ ] Calculate correct 8-byte padded alignments dynamically
- [ ] Write header size unsigned 64-bit integer
- [ ] Append binary buffers efficiently using `writev` / vectorized I/O
- [ ] Prevent memory explosion during serialization by streaming arrays sequentially
- [ ] Support generating `__metadata__` standard dictionary fields
- [ ] Support appending format version identifiers natively
- [ ] Support `safetensors.save_file` API parity
- [ ] Support `safetensors.save` (return raw bytes) API parity
- [ ] Warn against duplicate keys during generation
- [ ] Handle `bfloat16` generation securely
- [ ] Export massive 100GB+ arrays by creating chunked sharded sets automatically
- [ ] Generate the corresponding `model.safetensors.index.json` natively
- [ ] Validate generated files immediately via loopback reading
- [ ] Validate generated files are byte-for-byte identical to Rust reference implementation
- [ ] Stream serialization natively via `yield` buffers (chunked HTTP uploads)

### 9. Edge Cases, Framework Interop & Testing (30+ items)

- [ ] Unit Test: 0-byte tensor saving/loading (`shape=[]`)
- [ ] Unit Test: 1D, 2D, 3D, 4D standard Float32 arrays
- [ ] Unit Test: Endianness conversion tests natively
- [ ] Unit Test: JSON header > 10MB test (massive vocabulary models)
- [ ] Unit Test: Extremely high precision dimensions (e.g. `2^31 - 1`)
- [ ] Interop: Support loading PyTorch Safetensors correctly mapped to ONNX conventions
- [ ] Interop: Support loading TensorFlow Safetensors correctly mapped
- [ ] Interop: Support loading Flax/JAX Safetensors natively
- [ ] Remap Flax hierarchical keys (`layers.0.attention.kernel`) to standard `.weight` suffixes dynamically
- [ ] Provide strict dictionary equivalence testing (`np.testing.assert_allclose`)
- [ ] Test memory leak protections (asserting `mmap` references drop to 0)
- [ ] Validate Javascript WebAssembly Out-of-Bounds protections
- [ ] Expose benchmarking scripts comparing pure Python vs `pickle` vs `rust-safetensors`

### 10. Explicit JavaScript / WASM Typed Array Mappings (30+ items)

- [ ] Map Safetensors `F64` directly to JS `Float64Array` without duplication
- [ ] Map Safetensors `F32` directly to JS `Float32Array` without duplication
- [ ] Map Safetensors `I32` directly to JS `Int32Array` without duplication
- [ ] Map Safetensors `I16` directly to JS `Int16Array` without duplication
- [ ] Map Safetensors `I8` directly to JS `Int8Array` without duplication
- [ ] Map Safetensors `U32` directly to JS `Uint32Array` without duplication
- [ ] Map Safetensors `U16` directly to JS `Uint16Array` without duplication
- [ ] Map Safetensors `U8` directly to JS `Uint8Array` without duplication
- [ ] Map Safetensors `I64` safely to JS `BigInt64Array` natively
- [ ] Map Safetensors `U64` safely to JS `BigUint64Array` natively
- [ ] Emulate `F16` in JS using `Uint16Array` views (providing decoding hooks to `Float32`)
- [ ] Emulate `BF16` in JS using `Uint16Array` views (providing left-shift decoding hooks)
- [ ] Ensure JS TypedArrays are generated using `buffer.slice()` only if explicitly requested (memory copy)
- [ ] Default JS TypedArrays to `new Float32Array(buffer, byteOffset, length)` (zero-copy)
- [ ] Validate JS byte offsets are multiples of the TypedArray element sizes (padding checks)
- [ ] Provide unaligned buffer fallback (copying unaligned data to aligned arrays if zero-copy fails)
- [ ] Expose `SharedArrayBuffer` mapping for multi-threaded WebWorker access
- [ ] Pass Safetensors pointers directly into Pyodide WASM memory (`Module.HEAPU8`)
- [ ] Prevent JS Garbage Collector from sweeping the underlying ArrayBuffer prematurely
- [ ] Support Node.js `Buffer` natively without invoking browser-only APIs
- [ ] Support Node.js `fs.openSync` and `fs.readSync` for manual chunk reading
- [ ] Expose a Javascript generator `async function* load_tensors(file)`
- [ ] Parse UTF-8 JSON headers natively using JS `TextDecoder` (handling streaming bytes)

### 11. Comprehensive Error Handling & Exceptions (30+ items)

- [ ] Raise `SafetensorsHeaderTooLargeError` if header size `N` > 100MB
- [ ] Raise `SafetensorsInvalidHeaderError` if UTF-8 JSON decoding fails
- [ ] Raise `SafetensorsInvalidJSONError` if JSON parses but is structurally invalid
- [ ] Raise `SafetensorsDuplicateKeyError` if tensor names overlap
- [ ] Raise `SafetensorsInvalidOffsetError` if `begin` > `end`
- [ ] Raise `SafetensorsOutOfBoundsError` if `end` > `file_size`
- [ ] Raise `SafetensorsOverlapError` if data regions mathematically intersect
- [ ] Raise `SafetensorsAlignmentError` if `begin` is not 8-byte aligned
- [ ] Raise `SafetensorsInvalidDtypeError` if the `dtype` string is unrecognized
- [ ] Raise `SafetensorsShapeMismatchError` if `(end - begin) != volume(shape) * dtype_size`
- [ ] Raise `SafetensorsFileEmptyError` if file size is 0
- [ ] Raise `SafetensorsFileTooSmallError` if file size < 8 bytes
- [ ] Catch `OSError` / `IOError` cleanly during `mmap` initialization
- [ ] Catch `MemoryError` dynamically if system RAM cannot map the file (32-bit limits)
- [ ] Catch `RangeError` in Javascript if TypedArray bounds are exceeded
- [ ] Catch `TypeError` in Python if passing non-string keys to the dictionary interface
- [ ] Raise `SafetensorsWriteError` if serialization disk space is exhausted
- [ ] Provide explicit error boundaries for cross-platform file locking (Windows vs Linux)
- [ ] Provide explicit warnings if `__metadata__` is missing standard HuggingFace keys
- [ ] Gracefully catch and report JSON deeply nested recursion limits
- [ ] Ensure all custom exceptions subclass a base `SafetensorsError` for easy `try/except` handling

### 12. Hugging Face Hub Integration & Ecosystem (25+ items)

- [ ] Support direct parsing of `hf://` protocol URIs natively
- [ ] Authenticate HTTP Range requests using `HF_TOKEN` environment variables implicitly
- [ ] Parse Hugging Face Hub `model.safetensors.index.json` natively
- [ ] Resolve sharded file paths relative to the Hub repository structure
- [ ] Cache downloaded chunks dynamically to `~/.cache/huggingface/hub/` automatically
- [ ] Emulate `huggingface_hub` `cached_download` paths if the library is not installed
- [ ] Validate Hub ETag headers before resuming interrupted Range requests
- [ ] Warn on Hub Rate Limiting (HTTP 429) dynamically
- [ ] Back-off dynamically based on Hub `Retry-After` headers
- [ ] Verify Hub downloaded `.safetensors` against `sha256` hashes provided in the repository
- [ ] Support fetching weights directly from specific commits/branches natively (`revision=...`)
- [ ] Expose progress tracking callbacks compatible with standard `tqdm` bars
- [ ] Auto-detect if a model repository defaults to `.bin` (PyTorch) vs `.safetensors` and prioritize `.safetensors`
- [ ] Support PyTorch `load_state_dict` direct emulation (returning dict of PyTorch-compatible tensors)

### 13. Deep Framework Weight Layout Mappings (20+ items)

- [ ] Emulate PyTorch `Conv1d` weight layouts seamlessly (translating ONNX shapes if necessary)
- [ ] Emulate PyTorch `Conv2d` weight layouts seamlessly (`[O, I, H, W]`)
- [ ] Emulate PyTorch `Linear` weight layouts seamlessly (`[O, I]`)
- [ ] Emulate TensorFlow `Conv2D` weight layouts seamlessly (`[H, W, I, O]`)
- [ ] Emulate TensorFlow `Dense` weight layouts seamlessly (`[I, O]`)
- [ ] Emulate Flax `Dense` weight layouts seamlessly (`[I, O]`)
- [ ] Emulate Flax `Conv` weight layouts seamlessly (`[H, W, I, O]`)
- [ ] Provide dynamic transposition hooks: `tensor.transpose_on_load()`
- [ ] Support `Safetensor` weights that natively bake-in `GroupNormalization` scales/biases
- [ ] Support `Safetensor` weights mapped to `LayerNormalization` arrays
- [ ] Resolve QKV (Query/Key/Value) weight concatenation differences across PyTorch and TF natively
- [ ] Split loaded QKV tensors automatically if ONNX topological inputs expect separated Q, K, V
- [ ] Concatenate separated Q, K, V tensors automatically if ONNX topology expects a packed QKV

### 14. Performance Profiling & Advanced Benchmarking (20+ items)

- [ ] Benchmark: Peak memory usage loading 7B parameter model (should be ~0 RAM overhead via mmap)
- [ ] Benchmark: Total time to extract 10,000 specific small tensors from a massive file
- [ ] Benchmark: Total time to stream a single 1GB layer over HTTP (measuring overhead)
- [ ] Profile OS Page Cache hit-rates natively (if tooling allows)
- [ ] Profile Garbage Collection pressure in V8/Node.js after loading a 2GB model
- [ ] Unit Test: Concurrency (Read the same `.safetensors` file from 16 Threads simultaneously)
- [ ] Unit Test: Multiprocessing (Read the same `.safetensors` file from 4 Processes simultaneously)
- [ ] Unit Test: Write a 1GB `.safetensors` file and verify throughput > 500MB/s
- [ ] Unit Test: Load a completely sparse (all zeros) tensor array flawlessly
- [ ] Validate Python `memoryview` slice extraction latency (<1ms per slice)
- [ ] Monitor and test HTTP Keep-Alive connection limits natively to prevent socket exhaustion

### 15. WASM Specific Memory Alignment & Buffers (20+ items)

- [ ] Explicitly pad `U8` JS extracts to 8-byte boundaries if passing to WebAssembly
- [ ] Explicitly pad `I8` JS extracts to 8-byte boundaries if passing to WebAssembly
- [ ] Pad `F16` JS extracts to 8-byte boundaries natively before Emscripten ingestion
- [ ] Implement Emscripten `_malloc` wrapper in JS to pre-allocate exact payload sizes safely
- [ ] Handle Javascript `DataView` limits if `buffer.byteLength` > 2GB (Chrome/V8 limits)
- [ ] Ensure `Float32Array` views strictly align on 4-byte boundaries (JS spec)
- [ ] Ensure `Float64Array` views strictly align on 8-byte boundaries (JS spec)
- [ ] Copy unaligned `Float32` data explicitly (via `Uint8Array`) to aligned buffers if needed
- [ ] Fallback gracefully when `SharedArrayBuffer` is blocked by Cross-Origin-Opener-Policy (COOP) headers
- [ ] Leverage WebCodecs `VideoFrame` (hack) for GPU zero-copy upload if WebGL doesn't natively support TypedArrays
- [ ] Expose WASM linear memory bounds dynamically (`wasmMemory.buffer.byteLength`)
- [ ] Catch WASM `OOM` errors internally when Safetensors payload exceeds available Pages
- [ ] Trigger explicit JS GC (`gc()`) dynamically after offloading large tensors to WebGPU buffers

### 16. Advanced Dict Manipulation & Utility API (20+ items)

- [ ] Support PyTorch `state_dict()` semantic patching dynamically
- [ ] Rename keys natively during loading (`safetensors.load_file(..., prefix="model.")`)
- [ ] Filter keys natively using Regex (`safetensors.load_file(..., pattern=".*weight$")`)
- [ ] Merge two `.safetensors` files in-memory natively (`dict.update()`)
- [ ] Merge a `.bin` PyTorch checkpoint with a `.safetensors` dictionary visually
- [ ] Serialize merged dictionaries efficiently back to disk
- [ ] Extract single specific key natively: `safetensors.get_tensor("file.safetensors", "key")`
- [ ] Expose `save_file` parameter to explicitly overwrite existing files vs appending
- [ ] Output raw JSON dictionary explicitly: `safetensors.get_metadata("file.safetensors")`
- [ ] Verify file integrity without loading tensors: `safetensors.check_file_validity()`
- [ ] Provide utility to convert PyTorch `.bin` (Pickle) directory to `.safetensors` directory automatically
- [ ] Provide utility to convert TensorFlow `SavedModel` variables directly to `.safetensors`

### 17. Model-Specific Parsing Architectures & Edge Cases (15+ items)

- [ ] Parse explicitly LLaMA format Safetensors layouts (`layers.0.self_attn.q_proj.weight`)
- [ ] Parse explicitly BERT format Safetensors layouts (`bert.encoder.layer.0.attention.self.query.weight`)
- [ ] Parse explicitly Whisper format Safetensors layouts (encoder and decoder sub-dictionaries)
- [ ] Parse explicitly Stable Diffusion formats (U-Net, VAE, TextEncoder sharded safetensors)
- [ ] Parse SDXL massive UNet `.safetensors` dynamically
- [ ] Correctly handle empty `__metadata__` dictionaries (e.g. `{}`)
- [ ] Correctly handle `__metadata__` with explicit format strings (`"format": "pt"`)
- [ ] Extract HuggingFace specific quantization metadata (e.g. `bitsandbytes` scale parameters hidden in JSON)
- [ ] Verify `int4` block scaling arrays are mapped correctly relative to the primary weight

### 18. Final Precision, Testing, and Compliance Verification (15+ items)

- [ ] Profile HTTP Request times dynamically for 100 concurrent Range requests
- [ ] Measure total elapsed time parsing exactly 10,000 JSON keys natively
- [ ] Unit Test: Verify `bfloat16` mathematical conversions round accurately in JS
- [ ] Unit Test: Verify 1D `Int8` arrays load perfectly without padding issues
- [ ] Output `__metadata__` length natively before parsing tensors (for early exit checks)
- [ ] Log precise byte ranges applied during HTTP chunks
- [ ] Catch exactly `404 Not Found` for missing shards immediately
- [ ] Catch exactly `403 Forbidden` for private HuggingFace Repos without `HF_TOKEN`
- [ ] Guarantee no usage of `eval()` or `Function()` in JS parser natively
- [ ] Guarantee no usage of `eval()` or `exec()` in Python parser natively
- [ ] Export TypeScript `.d.ts` module securely exposing all API functions
