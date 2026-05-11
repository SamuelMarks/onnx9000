# MXNet Support

ONNX9000 supports importing legacy MXNet `-symbol.json` and `.params` files directly into the Core IR without requiring a working MXNet or Apache TVM installation.

## Supported Layers

- `Convolution`
- `Pooling`
- `Activation` (ReLU)
- `FullyConnected`
- Identity mappings for other types

## Usage

```bash
onnx9000 convert model-symbol.json --from mxnet --to onnx --weights model.params -o model.onnx
```
