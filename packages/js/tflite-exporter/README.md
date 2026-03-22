# @onnx9000/tflite-exporter

This module serves as a 100% dependency-free compiler designed to translate ONNX Intermediate Representations natively into Google's `.tflite` binary schema format and the TensorFlow SavedModel Protobuf. It is compatible with both Web V8 Javascript contexts, Mobile environments, and Python natively.

## Architecture and Scope

Instead of bridging out to multi-gigabyte `tensorflow` C++ logic, this package leverages native Web-First AST Graph transformations. Key architectural tasks include:

- Strict NCHW to NHWC topology transposition logic (baked directly at compile time).
- Explicit `Padding` emulation logic maintaining TFLite NNAPI fallback safety limits.
- FlatBuffer Builder execution (byte-for-byte compatible with Google's `flatc`).
- Asymmetric INT8 and FLOAT16 scale parameter extraction avoiding complicated post-training calibration dependencies.

## Compiling ONNX for Coral EdgeTPU via the Browser

This compiler isolates tensor dependencies securely, mapping specifically against Google's Coral DSP constraints. By pushing Transpose ops and statically calculating padding logic `[0,2,3,1]` mappings natively we assure that `EdgeTPU` runs fully mapped integer-based schemas flawlessly.

Check the `apps/demo-tflite-converter` UI to load a model and output standard `EdgeTPU` compatible binaries by merely selecting the Int8 checkmark. The UI utilizes background `WebWorkers` ensuring massive model allocations bypass blocking browser UI limits.

## Deploying ONNX models to Android using `onnx9000`

You can natively convert ONNX models into `TFLite` files ready for dropping directly into Android Studio / Java Native Inference contexts:

```bash
# via CLI (Outputs explicitly mapped mobilenet.tflite)
onnx9000 onnx2tf mobilenet.onnx --fp16 -o mobilenet.tflite
```

Because `tflite-exporter` supports explicitly mapped operations natively without Tensorflow bindings, your models will run standard Android NNAPI inference out-of-the-box natively mapped without custom operation logic.
