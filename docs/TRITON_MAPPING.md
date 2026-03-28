# ONNX to Triton Operator Mapping

This document outlines how `onnx9000` maps ONNX operators to OpenAI Triton (`@triton.jit`) Python code.

## Elementwise Operations

| ONNX Op | Triton Equivalent          |
| ------- | -------------------------- |
| Add     | `a + b`                    |
| Sub     | `a - b`                    |
| Mul     | `a * b`                    |
| Div     | `a / (b + 1e-10)`          |
| Exp     | `tl.exp(x)`                |
| Log     | `tl.log(x)`                |
| Sqrt    | `tl.sqrt(x)`               |
| Relu    | `tl.maximum(x, 0.0)`       |
| Sigmoid | `1.0 / (1.0 + tl.exp(-x))` |
| Tanh    | `tl.math.tanh(x)`          |

## Reductions

| ONNX Op   | Triton Equivalent      |
| --------- | ---------------------- |
| ReduceSum | `tl.sum(x, axis=0)`    |
| ReduceMax | `tl.max(x, axis=0)`    |
| ArgMax    | `tl.argmax(x, axis=0)` |

## Complex Layers

| Layer     | Implementation Strategy                                           |
| --------- | ----------------------------------------------------------------- |
| MatMul    | Tiled `tl.dot` with `BLOCK_K` accumulation loop.                  |
| LayerNorm | Two-pass reduction (mean, then variance).                         |
| Softmax   | Numerically stable `max` subtraction followed by `exp` and `sum`. |
