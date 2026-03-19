# Tutorial: Building a Zero-Dependency 10KB Image Classifier

This tutorial shows how to use `onnx9000-iree` to compile a standard ResNet or MobileNet model into a 10KB standalone JS file that uses WebGPU directly, bypassing heavy runtime libraries.

## 1. Get an ONNX Model
Download a standard ONNX vision model from the ONNX Model Zoo.

## 2. Compile to Standalone JS
Run the CLI command:
\`\`\`bash
npx @onnx9000/iree-compiler compile mobilenet.onnx --target-backend=standalone-js --optimize-level=O3
\`\`\`

## 3. Include in HTML
The compiler generates an `index.js` and `model.bin` containing the weights. It also produces a starter `index.html` template. Serve the directory with an HTTP server and open the browser. The inference will execute entirely on the GPU without dynamically parsing an AST.
