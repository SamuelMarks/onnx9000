# NCNN Support

ONNX9000 supports importing NCNN `.param` and `.bin` files directly into the Core IR.

## Supported Layers

- `Input`
- `Convolution`
- `ConvolutionDepthWise`
- `Pooling`
- `ReLU`
- `Split`

## Usage

```bash
onnx9000 convert model.param --from ncnn --to onnx --weights model.bin -o model.onnx
```
