---
orphan: true
---

# Sphinx Demo UI

The ONNX9000 Sphinx Demo UI is a fully interactive, in-browser execution experience seamlessly integrated into our Sphinx documentation. It allows users to drag-and-drop models, convert formats, and run inference using zero-dependency pure TypeScript and HTML.

## Features

- **No Frameworks:** Built entirely with vanilla TypeScript and HTML, avoiding React, Vue, or Angular to ensure maximum portability and minimal footprint.
- **WASM & WebGPU:** Executes models using native browser capabilities for near-native performance.
- **Polyglot Parsing:** Supports drag-and-drop ingestion of multiple model formats including ONNX, TensorFlow, Scikit-Learn, and PyTorch (via Pyodide).

## Command Line Interface (CLI)

You can launch a local standalone server for the Sphinx Demo UI using the following command:

```bash
onnx9000 sphinx-demo-ui
```

This will start a local HTTP server and open the interactive environment in your default web browser.
