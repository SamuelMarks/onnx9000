# CNTK Support

ONNX9000 supports importing legacy CNTK `.model` files (v2 protocol buffer format) directly into the Core IR without requiring a working CNTK installation.

## Supported Operations

- `Convolution`
- `Plus`
- Identity mappings for other types

## Dynamic Sequence Axes

CNTK's dynamic sequence axes are naturally represented as dynamic dimensions (`-1`) in ONNX9000's Core IR.

## Usage

```bash
onnx9000 convert network.model --from cntk --to onnx -o network.onnx
```
