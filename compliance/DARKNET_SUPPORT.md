# Darknet Support

ONNX9000 supports importing Darknet `.cfg` and `.weights` files directly to the Core IR.

## Supported Layers

- `[net]`
- `[convolutional]`
- `[route]`
- `[shortcut]`
- `[maxpool]`
- `[yolo]` (Mapped as output)

## Supported Activations

- `linear`
- `leaky`
- `mish`

## Usage

```bash
onnx9000 convert yolov3.cfg --from darknet --to onnx --weights yolov3.weights -o yolov3.onnx
```
