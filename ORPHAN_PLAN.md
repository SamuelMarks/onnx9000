# ONNX9000 Exhaustive Orphan Features Resolution Plan (Hyper-Granular)

This document represents the ultimate, atomic roadmap to resolve all ecosystem asymmetries ("orphans") within the `onnx9000` monorepo. Every missing link between the Python SDK, TypeScript SDK, CLI, Web Demos, and E2E Testing suites has been broken down into PR-sized, actionable checkbox items.

---

## 1. The JS-to-Python Ports (Missing from Python SDK)

These frameworks have robust JS SDK implementations inside `packages/js/converters/src/mmdnn`, complete with JS unit tests, but are completely missing from the Python SDK and Python CLI.

- [x] **Darknet**
  - [x] **Core Parsing (`packages/python/onnx9000-converters/src/onnx9000/converters/darknet/`)**
    - [x] Create `__init__.py` to expose Darknet parser and importer APIs.
    - [x] Create `parser.py` to parse Darknet `.cfg` files (INI-style format) into Python dictionaries.
    - [x] Create `weights.py` to handle `struct` unpacking of binary `.weights` files.
    - [x] Create `mapper.py` to map `[convolutional]`, `[yolo]`, and `[route]` layers to ONNX IR.
  - [x] **Integration & CLI**
    - [x] Register `DarknetConverter` in `packages/python/onnx9000-converters/src/onnx9000/converters/__init__.py`.
    - [x] Add `darknet` to `--from` choices in `apps/cli/src/onnx9000_cli/main.py`.
    - [x] Add CLI validation requiring both `--model` (for `.cfg`) and a new `--weights` argument.
  - [x] **Testing (`packages/python/onnx9000-converters/tests/darknet/`)**
    - [x] Create `test_parser.py` with mock `.cfg` strings to verify correct dictionary generation.
    - [x] Create `test_mapper.py` with isolated unit tests for layer-to-IR translations.
    - [x] Create `test_integration.py` to run full `.cfg` + `.weights` to ONNX conversions.
  - [x] **Documentation**
    - [x] Write `compliance/DARKNET_SUPPORT.md` outlining supported YOLO architectures.

- [x] **NCNN**
  - [x] **Core Parsing (`packages/python/onnx9000-converters/src/onnx9000/converters/ncnn/`)**
    - [x] Create `__init__.py` to expose NCNN module.
    - [x] Create `parser.py` to parse NCNN `.param` magic numbers and layer definitions.
    - [x] Create `weights.py` to load `.bin` float16/float32 blobs.
    - [x] Create `mapper.py` to map NCNN specific layer ops (e.g., `ConvolutionDepthWise`) to ONNX IR.
  - [x] **Integration & CLI**
    - [x] Register `NCNNConverter` in converter registry.
    - [x] Add `ncnn` to `--from` choices in CLI.
    - [x] Add CLI file extension validation for `.param` and `.bin`.
  - [x] **Testing (`packages/python/onnx9000-converters/tests/ncnn/`)**
    - [x] Create `test_parser.py` verifying magic number validation and text parsing.
    - [x] Create `test_mapper.py`.
    - [x] Create `test_integration.py` using dummy NCNN graphs.
  - [x] **Documentation**
    - [x] Write `compliance/NCNN_SUPPORT.md`.

- [x] **Caffe**
  - [x] **Core Parsing (`packages/python/onnx9000-converters/src/onnx9000/converters/caffe/`)**
    - [x] Create `__init__.py`.
    - [x] Create `parser.py` implementing a zero-dependency Protobuf text parser for `.prototxt`.
    - [x] Create `weights.py` implementing a zero-dependency Protobuf binary parser for `.caffemodel`.
    - [x] Create `mapper.py` handling Caffe's implicit batch dimensions and legacy NCHW layouts.
  - [x] **Integration & CLI**
    - [x] Register `CaffeConverter` in registry.
    - [x] Add `caffe` to `--from` choices in CLI.
  - [x] **Testing (`packages/python/onnx9000-converters/tests/caffe/`)**
    - [x] Create `test_prototxt_parser.py`.
    - [x] Create `test_mapper.py` focusing on `InnerProduct` and `Pooling` layers.
    - [x] Create `test_snapshots.py` targeting `snapshots/caffe-0.1.0.json`.
  - [x] **Documentation**
    - [x] Write `compliance/CAFFE_SUPPORT.md`.

- [x] **CNTK**
  - [x] **Core Parsing (`packages/python/onnx9000-converters/src/onnx9000/converters/cntk/`)**
    - [x] Create `parser.py` for reading CNTK v2 Protocol Buffer `.model` files.
    - [x] Create `mapper.py` for handling CNTK's dynamic sequence axes.
  - [x] **Integration & Testing**
    - [x] Register `CNTKConverter` and add `cntk` to CLI.
    - [x] Add `test_parser.py` and `test_mapper.py` in `tests/cntk/`.
    - [x] Verify against `snapshots/cntk-2.7.post2.json`.
  - [x] **Documentation**
    - [x] Write `compliance/CNTK_SUPPORT.md`.

- [x] **MXNet**
  - [x] **Core Parsing (`packages/python/onnx9000-converters/src/onnx9000/converters/mxnet/`)**
    - [x] Create `parser.py` for loading MXNet `-symbol.json` graphs.
    - [x] Create `weights.py` for parsing NDArray `.params` binary structures.
    - [x] Create `mapper.py`.
  - [x] **Integration & Testing**
    - [x] Register `MXNetConverter` and add `mxnet` to CLI.
    - [x] Add `test_parser.py` and `test_mapper.py` in `tests/mxnet/`.
    - [x] Add snapshot validation using `snapshots/mxnet-1.9.1.json`.

---

## 2. The Python-to-JS Ports (Missing from JS SDK)

These frameworks have comprehensive Python SDK implementations but are completely missing from the JS SDK, rendering them unavailable for in-browser execution.

- [x] **JAX / Flax**
  - [x] **JS Parsing (`packages/js/converters/src/jax/`)**
    - [x] Create `index.ts` to export JAX converter utilities.
    - [x] Create `jaxpr_parser.ts` to parse raw stringified `jaxpr` JSON.
    - [x] Create `flax_parser.ts` to handle Flax `nnx` module state dictionaries.
    - [x] Create `mapper.ts` to translate XLA/HLO operations to ONNX IR.
  - [x] **JS Testing (`packages/js/converters/tests/jax/`)**
    - [x] Create `jaxpr_parser.test.ts` to verify AST generation.
    - [x] Create `mapper.test.ts` to verify math op translation.
  - [x] **Web Integration**
    - [x] Export JAX modules in `packages/js/converters/src/index.ts`.
    - [x] Update `frameworkRequirements` in `apps/demo-mmdnn/src/main.ts` to accept `.json` (jaxpr text).
    - [x] Update HTML dropdown in `apps/demo-mmdnn/index.html` to include "JAX/Flax".

- [x] **H2O**
  - [x] **JS Parsing (`packages/js/converters/src/mmdnn/h2o/`)**
    - [x] Create `index.ts`.
    - [x] Create `parser.ts` ported from `h2o.py` to parse H2O MOJO models.
    - [x] Create `mapper.ts` to translate MOJO trees to `TreeEnsembleRegressor`.
  - [x] **JS Testing**
    - [x] Create `packages/js/converters/tests/mmdnn/h2o/parser.test.ts`.
  - [x] **Web Integration**
    - [x] Update `apps/demo-mmdnn/src/main.ts` to support `.zip` (MOJO) files for H2O.
    - [x] Add H2O to Web UI framework dropdowns.

- [x] **LibSVM**
  - [x] **JS Parsing (`packages/js/converters/src/mmdnn/libsvm/`)**
    - [x] Create `parser.ts` ported from `libsvm.py` (parsing text-based SVM formats).
    - [x] Create `mapper.ts` mapping to `SVMClassifier` / `SVMRegressor`.
  - [x] **JS Testing**
    - [x] Create `packages/js/converters/tests/mmdnn/libsvm/parser.test.ts`.
  - [x] **Web Integration**
    - [x] Add `.svm` and `.txt` file support for LibSVM in `apps/demo-mmdnn/src/main.ts`.

---

## 3. The "Wiring" Orphans (Missing CLI / Web Hooks)

These traditional ML frameworks are fully implemented in BOTH Python and JS SDKs but have been neglected in user-facing applications (CLI and Web UIs).

- [x] **Scikit-Learn (skl2onnx)**
  - [x] **CLI Wiring (`apps/cli/src/onnx9000_cli/main.py`)**:
    - [x] Add `"sklearn"` to `--from` `choices` list.
    - [x] Add dispatch logic to `convert_cmd`: `if args.from_fmt == "sklearn": ...`
    - [x] Implement a safe `joblib` unpickler warning or direct JSON pipeline schema reader.
  - [x] **Web UI Wiring (`apps/demo-mmdnn/src/main.ts`)**:
    - [x] Add `sklearn` to `frameworkRequirements` (Requires: `.joblib` or `.json`).
    - [x] Add `<option value="sklearn">Scikit-Learn</option>` to `index.html`.

- [x] **XGBoost**
  - [x] **CLI Wiring**: Add `"xgboost"` to CLI `--from` choices and route to `parse_xgboost_json`.
  - [x] **Web UI Wiring**: Add `xgboost` (Requires: `.json` dump) to `demo-mmdnn` dropdowns.

- [x] **CatBoost**
  - [x] **CLI Wiring**: Add `"catboost"` to CLI `--from` choices and route to `parse_catboost_json`.
  - [x] **Web UI Wiring**: Add `catboost` (Requires: `.json` dump) to `demo-mmdnn` dropdowns.

- [x] **LightGBM**
  - [x] **Documentation**: Write the missing `compliance/LIGHTGBM_SUPPORT.md`.
  - [x] **CLI Wiring**: Add `"lightgbm"` to CLI `--from` choices.
  - [x] **Web UI Wiring**: Add `lightgbm` to `demo-mmdnn` dropdowns.

- [x] **SparkML (PySpark)**
  - [x] **CLI Wiring**: Add `"pyspark"` to CLI `--from` choices.
  - [x] **Web UI Wiring**: Add `pyspark` to `demo-mmdnn` dropdowns.

- [x] **PaddlePaddle**
  - [x] **CLI Wiring**: Add `"paddle"` to CLI `--from` choices. (Already exists in Web UI).

---

## 4. UI Testing Orphans (Missing E2E Playwright Coverage)

These interactive Web UIs exist in `apps/` but have 0% E2E test coverage in the `e2e/` folder.

- [x] **demo-tflite-converter (`e2e/demo-tflite-converter.spec.ts`)**
  - [x] Write test: Page loads without console errors.
  - [x] Write test: Upload a dummy `.onnx` file via the file input.
  - [x] Write test: Click "Convert to TFLite" button.
  - [x] Write test: Intercept WASM compilation success and verify download blob is triggered.

- [x] **netron-ui (`e2e/netron-ui.spec.ts`)**
  - [x] Write test: Page loads and `<canvas id="graph-canvas">` is present.
  - [x] Write test: Upload a minimal ONNX graph.
  - [x] Write test: Click a node in the canvas and verify the right-side properties panel opens.
  - [x] Write test: Verify mouse-wheel zooming updates canvas transform.

- [x] **onnx-checker-ui (`e2e/onnx-checker-ui.spec.ts`)**
  - [x] Write test: Upload a known-valid ONNX file and verify "Validation Passed" UI state.
  - [x] Write test: Upload a known-invalid ONNX file (e.g. missing inputs) and verify error messages are rendered in the DOM.

- [x] **onnx2c-ui (`e2e/onnx2c-ui.spec.ts`)**
  - [x] Write test: Upload ONNX file.
  - [x] Write test: Verify C99 code generation completes.
  - [x] Write test: Assert the output `<pre>` tag contains `#include <stdio.h>` and `float* outputs`.

- [x] **onnx2gguf-ui (`e2e/onnx2gguf-ui.spec.ts`)**
  - [x] Write test: Verify UI allows uploading `tokenizer.json` alongside the `.onnx` model.
  - [x] Write test: Select "Q4_K_M" from the quantization dropdown.
  - [x] Write test: Click convert and verify UI progress bar updates.

- [x] **openvino-ui (`e2e/openvino-ui.spec.ts`)**
  - [x] Write test: Upload ONNX file.
  - [x] Write test: Trigger OpenVINO export.
  - [x] Write test: Verify UI offers a `.zip` download containing `.xml`, `.bin`, and `.mapping` files.

- [x] **optimum-ui (`e2e/optimum-ui.spec.ts`)**
  - [x] Write test: Enter a mock HuggingFace model ID in the text field.
  - [x] Write test: Select optimization level "O3".
  - [x] Write test: Click "Optimize" and mock the network response.

- [x] **sphinx-demo-ui (`e2e/sphinx-demo-ui.spec.ts`)**
  - [x] Write test: Load the mock documentation page.
  - [x] Write test: Verify the embedded interactive API playground elements are clickable and responsive.
