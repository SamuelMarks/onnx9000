# H2O Support

ONNX9000 provides zero-dependency parsing of H2O MOJO models directly into the ONNX IR using pure Python and TypeScript.

## Supported Formats

- `.zip` (H2O MOJO export containing model details and tree structures)

## Supported Algorithms

The following H2O algorithms are supported and mapped to ONNX operators:

- **Distributed Random Forest (DRF)** -> `TreeEnsembleClassifier` / `TreeEnsembleRegressor`
- **Gradient Boosting Machine (GBM)** -> `TreeEnsembleClassifier` / `TreeEnsembleRegressor`
- **XGBoost (via H2O)** -> `TreeEnsembleClassifier` / `TreeEnsembleRegressor`
- **Deep Learning** -> `MatMul` + `Add` + Activations (Mapped dynamically based on architecture)

## Usage

### CLI Usage

```bash
# Convert a MOJO zip file to ONNX
onnx9000 convert model.zip --from h2o --to onnx -o h2o_model.onnx
```

### Web UI

H2O MOJO models can be directly uploaded using the Universal Converter demo (`/mmdnn`), selecting "H2O MOJO" as the source framework.
