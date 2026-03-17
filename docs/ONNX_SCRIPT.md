# onnx9000.script Documentation

`onnx9000.script` provides a fluent, pure-Python authoring environment for ONNX, similar to Microsoft's ONNXScript and Quantco's Spox, but built entirely without the C++ `protobuf` extension. This makes it perfect for generating ONNX models dynamically directly in browser environments (like Pyodide/WASM).

## Features

- **Dynamic Op Namespace**: Use `op.Add(A, B)` or `op.Relu(X)`.
- **Operator Overloading**: Use standard Python operators (`A + B`, `A * B`, `A > B`).
- **Control Flow**: Supports mapping `if`, `for`, and `while` statements natively into ONNX `If` and `Loop` subgraphs.
- **Type Annotations**: Annotate inputs to strictly bind them to graph `ValueInfoProto`.

## Usage

```python
from onnx9000.script import script, op
from onnx9000.core.dtypes import DType

@script
def my_model(x):
    return op.Relu(x + 1)

model_proto = my_model.to_builder().to_onnx()
```
