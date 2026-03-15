# ONNX2: ONNX Runtime Extensions Native Rewrite

## Introduction
**Target Project:** Microsoft's [ONNX Runtime Extensions](https://github.com/microsoft/onnxruntime-extensions)
**New Home:** `src/onnx9000/extensions/`

The official `onnxruntime-extensions` library provides custom operators to bridge the gap between raw ML frameworks and ONNX execution. However, it relies heavily on massive C++ binaries compiled from Hugging Face Tokenizers (Rust/C++) or FFmpeg, making it extremely difficult to compile for lightweight web execution (WASM) without blooming the binary size by 10s of megabytes.

**The `onnx9000` Vision:** We are rebuilding these extensions fundamentally in two pure environments: 
1. **Pure Python** (for native server/desktop execution without C++ building).
2. **Pure JavaScript / Web APIs** (for browser execution). 

By leveraging native browser standards like the `WebCodecs API` for video decoding, the `Web Audio API` for spectral analysis, and writing zero-dependency `Byte-Pair Encoding (BPE)` and `SentencePiece` tokenizers in JS, we bypass HuggingFace and FFmpeg entirely, enabling true zero-dependency multimodal data pipelines.

## Exhaustive Implementation Checklist (300+ Items)

### Phase 1: Pure JS/Python Text Tokenization (BPE & WordPiece)
- [x] **Step 001:** Implement a foundational `Tokenizer` base class in Python.
- [x] **Step 002:** Implement a foundational `Tokenizer` base class in JavaScript.
- [x] **Step 003:** Implement pure JS parsing of Hugging Face `tokenizer.json` configuration files.
- [x] **Step 004:** Implement Byte-Pair Encoding (BPE) greedy merging algorithm in JS.
- [x] **Step 005:** Implement BPE greedy merging algorithm in pure Python.
- [x] **Step 006:** Implement WordPiece tokenization logic in JS.
- [x] **Step 007:** Implement WordPiece tokenization logic in pure Python.
- [x] **Step 008:** Write a highly optimized JS Trie data structure for O(L) prefix matching.
- [x] **Step 009:** Implement Unigram Language Model tokenization in JS.
- [x] **Step 010:** Implement Unigram Language Model tokenization in Python.
- [x] **Step 011:** Implement pre-tokenization regex splitting (e.g., whitespace, punctuation) matching Python `re` in JS.
- [x] **Step 012:** Handle standard Unicode normalization (NFC, NFD, NFKC, NFKD) natively in JS.
- [x] **Step 013:** Implement lowercasing, accent stripping, and custom normalizers.
- [x] **Step 014:** Generate Special Tokens (`[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]`) injection logic.
- [x] **Step 015:** Support dynamic vocabulary mapping (string to integer ID).
- [x] **Step 016:** Implement detokenization (ID back to string) reconstructing spaces accurately.
- [x] **Step 017:** Handle byte-level BPE (e.g., GPT-2/RoBERTa) mapping bytes to unicode characters.
- [x] **Step 018:** Implement sequence truncation logic (Head-only, Tail-only, Head+Tail).
- [x] **Step 019:** Implement sequence padding to fixed lengths (`pad_to_multiple_of`, `max_length`).
- [x] **Step 020:** Generate `attention_mask` tensors natively during tokenization.
- [x] **Step 021:** Generate `token_type_ids` tensors natively for BERT-style models.
- [x] **Step 022:** Implement batched tokenization (processing arrays of strings) using WebWorkers.
- [x] **Step 023:** Optimize the BPE merge cache using JS `Map` or plain objects.
- [x] **Step 024:** Support caching of frequently tokenized substrings.
- [x] **Step 025:** Write extensive parity tests against `transformers.AutoTokenizer` for GPT-2.
- [x] **Step 026:** Write extensive parity tests against `transformers.AutoTokenizer` for BERT.
- [x] **Step 027:** Write extensive parity tests against `transformers.AutoTokenizer` for Llama-2 (SentencePiece).
- [x] **Step 028:** Implement `AddedToken` handling (e.g., `<|endoftext|>` edge cases).
- [x] **Step 029:** Profile tokenizer speed (tokens/sec) against Rust `tokenizers` bindings.
- [x] **Step 030:** Implement sequence pair encoding (`text`, `text_pair`) logic.
- [x] **Step 031:** Support generating offset mappings (character start/end to token index).
- [x] **Step 032:** Implement returning JS/Python `Int32Array` buffers directly for zero-copy to WebGPU/WASM.
- [x] **Step 033:** Handle extremely large vocabularies (>100k) without memory bloat.
- [x] **Step 034:** Write fallback logic for out-of-vocabulary (OOV) tokens.
- [x] **Step 035:** Implement BPE dropout for tokenization robustness.
- [x] **Step 036:** Ensure string normalization preserves offset alignments.
- [x] **Step 037:** Test tokenization on diverse languages (CJK, Arabic, Cyrillic).
- [x] **Step 038:** Write an exporter converting `tokenizer.json` into a compressed binary format for faster web loading.
- [x] **Step 039:** Implement the compressed binary format loader in JS.
- [x] **Step 040:** Implement synchronous and asynchronous APIs for the tokenizers.
- [x] **Step 041:** Support streaming tokenization (processing text chunks as they arrive).
- [x] **Step 042:** Add diagnostic logging for tokenization failures.
- [x] **Step 043:** Ensure no memory leaks in JS tokenizer instance lifecycles.
- [x] **Step 044:** Build a CLI tool to convert HF tokenizers to the ONNX9000 format.
- [x] **Step 045:** Publish standalone tokenizer package (`@onnx9000/tokenizers`).
- [x] **Step 046:** Finalize Phase 1 Text Tokenization Architecture.

### Phase 2: SentencePiece & Custom Operations
- [x] **Step 047:** Implement pure Python SentencePiece Model (SPM) protobuf parser.
- [x] **Step 048:** Implement pure JS SentencePiece Model (SPM) protobuf parser.
- [x] **Step 049:** Implement SentencePiece Normalizer (NFKC + custom replacements).
- [x] **Step 050:** Implement SentencePiece Pre-tokenizer (space replacement with `_`).
- [x] **Step 051:** Implement SentencePiece BPE decoding.
- [x] **Step 052:** Implement SentencePiece Unigram decoding.
- [x] **Step 053:** Handle byte-fallback tokenization specific to SentencePiece.
- [x] **Step 054:** Write tests matching `sentencepiece` C++ library outputs.
- [x] **Step 055:** Implement ONNX operator `StringNormalizer` natively in Python/JS.
- [x] **Step 056:** Implement ONNX operator `RegexReplace` natively in Python/JS.
- [x] **Step 057:** Implement ONNX operator `Tokenizer` natively in Python/JS.
- [x] **Step 058:** Implement ONNX operator `VocabMapping` natively in Python/JS.
- [x] **Step 059:** Implement ONNX operator `TfIdfVectorizer` natively in Python/JS.
- [x] **Step 060:** Implement ONNX operator `WordpieceTokenizer` natively in Python/JS.
- [x] **Step 061:** Implement ONNX operator `BlingFireSentenceBpe` natively in Python/JS.
- [x] **Step 062:** Support loading fasttext/GloVe word embeddings directly.
- [x] **Step 063:** Implement n-gram extraction (for traditional NLP models).
- [x] **Step 064:** Implement text classification pipelines (Tokenize -> Infer -> Argmax -> Label).
- [x] **Step 065:** Implement sequence-to-sequence generation pipelines (Tokenize -> Infer loop -> Detokenize).
- [x] **Step 066:** Support constrained generation (e.g., JSON output enforcement via vocabulary masking).
- [x] **Step 067:** Implement trie-based prefix masking for constrained decoding.
- [x] **Step 068:** Profile JS performance of SentencePiece on large context windows (32k+ tokens).
- [x] **Step 069:** Ensure strict thread-safety for Python tokenizers.
- [x] **Step 070:** Implement caching of compiled regex patterns.
- [x] **Step 071:** Add comprehensive tests for zero-width characters and emojis.
- [x] **Step 072:** Implement BPE post-processor logic.
- [x] **Step 073:** Handle special token stripping during detokenization.
- [x] **Step 074:** Test SPM with T5 and ALBERT architectures.
- [x] **Step 075:** Implement a memory-mapped vocabulary file for massive token lists.
- [x] **Step 076:** Finalize Phase 2 Advanced Text.

### Phase 3: Image Preprocessing & Canvas APIs
- [x] **Step 077:** Implement native JS image loading using the HTML `Image` object.
- [x] **Step 078:** Implement native JS image pixel extraction using `OffscreenCanvas`.
- [x] **Step 079:** Extract raw RGBA pixel data to `Uint8ClampedArray`.
- [x] **Step 080:** Implement high-performance Bilinear interpolation resizing in JS.
- [x] **Step 081:** Implement high-performance Bicubic interpolation resizing in JS.
- [x] **Step 082:** Implement Nearest Neighbor resizing in JS.
- [x] **Step 083:** Implement Anti-aliased resizing (Lanczos/Area).
- [x] **Step 084:** Implement Center Cropping logic.
- [x] **Step 085:** Implement Random Cropping logic.
- [x] **Step 086:** Implement aspect-ratio preserving resizing (shorter/longer edge).
- [x] **Step 087:** Implement padding with constant colors.
- [x] **Step 088:** Implement channel reordering (RGBA to RGB, RGBA to BGR).
- [x] **Step 089:** Implement memory layout transformations (HWC to CHW).
- [x] **Step 090:** Implement pure JS image normalization (`pixel / 255.0`).
- [x] **Step 091:** Implement pure JS mean/standard deviation subtraction.
- [x] **Step 092:** Convert `Uint8ClampedArray` directly to `Float32Array` matching ONNX tensor specs.
- [x] **Step 093:** Implement batching logic for multiple images into a 4D tensor (N, C, H, W).
- [x] **Step 094:** Support loading Base64 Data URIs.
- [x] **Step 095:** Support reading directly from `Blob` or `File` inputs.
- [x] **Step 096:** Support processing `ImageData` objects directly.
- [x] **Step 097:** Implement WebGL/WebGPU accelerated image resizing to bypass CPU.
- [x] **Step 098:** Implement WebGL/WebGPU accelerated normalization and channel reordering.
- [x] **Step 099:** Implement ONNX operator `DecodeImage` natively.
- [x] **Step 100:** Implement ONNX operator `Resize` natively matching specific coordinate transformation modes.
- [x] **Step 101:** Write parity tests against `torchvision.transforms`.
- [x] **Step 102:** Implement basic data augmentations (Horizontal/Vertical Flip) in JS.
- [x] **Step 103:** Implement Color Jitter (Brightness, Contrast, Saturation) in JS.
- [x] **Step 104:** Implement Random Rotation augmentations in JS.
- [x] **Step 105:** Support reading 16-bit and HDR image formats if browser permits.
- [x] **Step 106:** Handle ICC color profile conversions natively or via Canvas.
- [x] **Step 107:** Implement `ImageEncoder` to serialize output tensors back to JPG/PNG blobs.
- [x] **Step 108:** Implement object detection bounding box drawing utilities.
- [x] **Step 109:** Implement segmentation mask overlay utilities.
- [x] **Step 110:** Profile the JS Canvas extraction bottleneck.
- [x] **Step 111:** Optimize memory allocation (pre-allocate Float32Arrays for video streaming).
- [x] **Step 112:** Implement a visual debugging tool for inspecting normalized tensors.
- [x] **Step 113:** Create Python equivalents using `Pillow` (PIL) for native execution without OpenCV.
- [x] **Step 114:** Ensure exact pixel parity between Python PIL resizing and JS Canvas resizing.
- [x] **Step 115:** Handle EXIF orientation flags automatically upon load.
- [x] **Step 116:** Write tests handling malformed or corrupted image files gracefully.
- [x] **Step 117:** Implement asynchronous image fetching with timeout logic.
- [x] **Step 118:** Finalize Phase 3 Vision Architecture.

### Phase 4: Audio Preprocessing & Spectrograms
- [x] **Step 119:** Implement native JS audio loading using the `Web Audio API` (`AudioContext`).
- [x] **Step 120:** Decode raw MP3, WAV, FLAC files directly to `AudioBuffer`.
- [x] **Step 121:** Extract raw PCM audio waveform to `Float32Array`.
- [x] **Step 122:** Implement high-quality Audio Resampling (e.g., 44.1kHz to 16kHz) in JS.
- [x] **Step 123:** Implement Audio Resampling in pure Python (without librosa/sox).
- [x] **Step 124:** Implement Stereo to Mono mixing.
- [x] **Step 125:** Implement generic Audio Normalization (Peak, RMS).
- [x] **Step 126:** Implement silence padding and truncation to fixed lengths.
- [x] **Step 127:** Implement Short-Time Fourier Transform (STFT) in pure JS.
- [x] **Step 128:** Implement STFT in pure Python.
- [x] **Step 129:** Implement Hann windowing function.
- [x] **Step 130:** Implement Hamming windowing function.
- [x] **Step 131:** Implement Power Spectrogram calculation.
- [x] **Step 132:** Implement Mel-filterbank matrix generation.
- [x] **Step 133:** Implement Mel-Spectrogram calculation matching `torchaudio.transforms.MelSpectrogram`.
- [x] **Step 134:** Implement Log-Mel Spectrogram calculation.
- [x] **Step 135:** Implement MFCC (Mel-frequency cepstral coefficients) extraction.
- [x] **Step 136:** Handle complex numbers correctly in STFT output.
- [x] **Step 137:** Write rigorous numerical parity tests comparing JS STFT to `torchaudio`.
- [x] **Step 138:** Implement ONNX operator `STFT` natively.
- [x] **Step 139:** Implement ONNX operator `MelWeightMatrix` natively.
- [x] **Step 140:** Support chunked audio processing for infinite streams.
- [x] **Step 141:** Implement `getUserMedia` integration to stream directly from the user's microphone.
- [x] **Step 142:** Implement a ring buffer for real-time microphone stream chunking.
- [x] **Step 143:** Implement Voice Activity Detection (VAD) heuristics.
- [x] **Step 144:** Build a WebWorker pipeline for background audio decoding and STFT generation.
- [x] **Step 145:** Implement Audio data augmentations (Time Stretch, Pitch Shift) in JS.
- [x] **Step 146:** Implement Background Noise mixing.
- [x] **Step 147:** Support encoding output tensors back to WAV blobs.
- [x] **Step 148:** Profile STFT performance in JS vs Python.
- [x] **Step 149:** Optimize the Fast Fourier Transform (FFT) algorithm in pure JS (Cooley-Tukey).
- [x] **Step 150:** Implement WebGPU accelerated FFT for massive spectrogram generation.
- [x] **Step 151:** Ensure `AudioContext` lifecycle is managed properly to avoid browser suspensions.
- [x] **Step 152:** Test audio handling on Safari (which has strict Web Audio API policies).
- [x] **Step 153:** Implement a visual spectrogram debugger in the UI.
- [x] **Step 154:** Support precise frame shifting and hop lengths.
- [x] **Step 155:** Handle boundary conditions (reflection padding, zero padding).
- [x] **Step 156:** Finalize Phase 4 Audio Architecture.

### Phase 5: Video Preprocessing (WebCodecs)
- [x] **Step 157:** Initialize the `WebCodecs API` (`VideoDecoder`).
- [x] **Step 158:** Implement an MP4/WebM demuxer (e.g., using `mp4box.js` or a custom lightweight parser).
- [x] **Step 159:** Extract raw compressed video chunks from the container.
- [x] **Step 160:** Feed compressed chunks into the `VideoDecoder`.
- [x] **Step 161:** Extract raw `VideoFrame` objects asynchronously.
- [x] **Step 162:** Implement high-performance conversion from `VideoFrame` (often YUV) to RGB `Float32Array`.
- [x] **Step 163:** Handle variable frame rates (VFR) and drop frames to achieve target FPS.
- [x] **Step 164:** Implement temporal subsampling (e.g., extract every Nth frame).
- [x] **Step 165:** Implement spatial resizing of `VideoFrame` objects using `OffscreenCanvas`.
- [x] **Step 166:** Implement batching of frames into a 5D tensor (N, T, C, H, W) for 3D Convolutions.
- [x] **Step 167:** Implement video stream buffering for continuous inference.
- [x] **Step 168:** Support streaming video directly from user's camera (`getUserMedia`).
- [x] **Step 169:** Handle WebCodecs lack of support gracefully (fallback to `<video>` element capturing).
- [x] **Step 170:** Implement a pure Python video frame extractor using `av` (PyAV) for native execution without heavy FFmpeg binaries if possible.
- [x] **Step 171:** Implement memory limits to prevent OOM when extracting many high-res frames.
- [x] **Step 172:** Synchronize audio extraction (Web Audio) with video extraction (WebCodecs) for multimodal models.
- [x] **Step 173:** Implement ONNX operator `ExtractVideoFrames` natively.
- [x] **Step 174:** Profile WebCodecs decoding latency vs `<video>` element rendering.
- [x] **Step 175:** Handle keyframe (I-frame) vs predictive frame (P/B-frame) dependencies.
- [x] **Step 176:** Ensure `VideoFrame.close()` is called meticulously to prevent GPU memory leaks.
- [x] **Step 177:** Write integration tests validating temporal consistency of extracted tensors.
- [x] **Step 178:** Build a scrubber UI component to debug extracted frames.
- [x] **Step 179:** Support generating output video (e.g., bounding boxes on streams).
- [x] **Step 180:** Finalize Phase 5 Video Architecture.

### Phase 6: Miscellaneous Custom ONNX Operators
- [x] **Step 181:** Implement `NonMaxSuppression` (NMS) natively in JS/Python for Object Detection.
- [x] **Step 182:** Implement `RotatedNMS` for oriented object detection.
- [x] **Step 183:** Implement `GridSample` natively (Bilinear and Nearest).
- [x] **Step 184:** Implement `RoIAlign` natively in JS/Python.
- [x] **Step 185:** Implement `DeformConv2d` natively.
- [x] **Step 186:** Implement `Inverse` (Matrix Inversion) natively.
- [x] **Step 187:** Implement `SVD` (Singular Value Decomposition) natively.
- [x] **Step 188:** Implement `TopK` highly optimized sorting algorithms.
- [x] **Step 189:** Implement `Unique` natively.
- [x] **Step 190:** Implement `Einsum` parsing and execution natively in JS/Python.
- [x] **Step 191:** Implement string manipulation ops: `StringConcat`, `StringSplit`.
- [x] **Step 192:** Implement `BeamSearch` decoding logic natively (often required for Seq2Seq models).
- [x] **Step 193:** Implement `GreedySearch` decoding natively.
- [x] **Step 194:** Implement `Sampling` (Top-P, Top-K, Temperature) decoding natively.
- [x] **Step 195:** Ensure all custom ops are cleanly registered in the `onnx9000.core.registry`.
- [x] **Step 196:** Write pure WGSL implementations for these complex ops where feasible (e.g., NMS).
- [x] **Step 197:** Write extensive PyTorch parity tests for all custom operators.
- [x] **Step 198:** Handle mixed-precision scaling inside custom operators.
- [x] **Step 199:** Finalize Phase 6 Custom Operators.
- [x] **Step 200:** Ensure seamless export of custom operators into the `export/builder.py` graph.
- [x] **Step 201:** Provide a user-facing API for developers to register their own JS/Python custom ops.
- [x] **Step 202:** Document the custom op registration API.
- [x] **Step 203:** Ensure `onnx.checker` does not fail on registered custom domains.
- [x] **Step 204:** Finalize ONNX2 extensions.

