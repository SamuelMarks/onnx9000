# LibSVM Support

ONNX9000 supports parsing text-based LibSVM files and translating them into standard ONNX Support Vector Machine (SVM) operators.

## Supported Formats

- `.svm` or `.txt` (Text format based on LIBSVM standards)

## Supported Models

- **SVC (Support Vector Classification)** -> `SVMClassifier`
- **SVR (Support Vector Regression)** -> `SVMRegressor`

## Mapped Parameters

The following kernel parameters and attributes are extracted and mapped into the `SVMClassifier` or `SVMRegressor` node attributes:

- `kernel_type` (linear, poly, rbf, sigmoid)
- `rho`
- `gamma`
- `coef0`
- `degree`
- SV (Support Vectors extraction)

## Usage

### CLI Usage

```bash
# Convert a LibSVM text file to ONNX
onnx9000 convert model.svm --from libsvm --to onnx -o model.onnx
```

### Web UI

LibSVM models can be directly uploaded using the Universal Converter demo (`/mmdnn`), selecting "LibSVM" as the source framework.
