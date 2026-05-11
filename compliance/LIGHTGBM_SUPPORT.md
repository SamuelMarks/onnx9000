# LightGBM Support

ONNX9000 supports importing legacy LightGBM `.txt` or `.json` models directly into the Core IR.

## Supported Operations

- `TreeEnsembleRegressor`
- `TreeEnsembleClassifier`

## Usage

```bash
onnx9000 convert model.json --from lightgbm --to onnx -o model.onnx
```
