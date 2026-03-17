# ONNX Ops Supported by WebGPU Backend

The onnx9000 pure Python authoring API can generate any ONNX operator dynamically. However, when executing those models via the WebGPU backend in Pyodide, the following opset 18+ subset is supported:

## Mathematics

- Add
- Sub
- Mul
- Div
- Pow
- MatMul
- Gemm
- Neg

## Neural Network

- Relu
- Sigmoid
- Tanh
- Conv
- MaxPool
- AveragePool
- Softmax

## Tensor Operations

- Reshape
- Transpose
- Squeeze
- Unsqueeze
- Concat
- Slice
- Gather
- ScatterND

## Control Flow

- If
- Loop
- Scan

_Custom operators can be dynamically compiled to WGSL if a `@onnx9000.kernel` decorator is provided alongside the `@onnx9000.script` authoring logic._
