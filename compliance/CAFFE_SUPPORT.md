# Caffe Support

ONNX9000 supports importing legacy Caffe `.prototxt` and `.caffemodel` files directly into the Core IR without requiring a working Caffe installation.

## Supported Layers

- `Input`
- `Data`
- `Convolution`
- `InnerProduct`
- `Pooling`
- `ReLU`
- `Softmax`

## Usage

```bash
onnx9000 convert deploy.prototxt --from caffe --to onnx --weights weights.caffemodel -o model.onnx
```
