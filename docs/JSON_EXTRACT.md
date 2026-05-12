---
orphan: true
---

# JSON Extraction (`json-extract`)

ONNX9000 allows you to extract the full topology, metadata, and structural graph of an ONNX file into a standard, human-readable JSON format.

This functionality is available via the CLI, the Python SDK, the JS SDK, and an interactive Web Demo.

## Command Line Interface (CLI)

Use the `json-extract` command to parse an `.onnx` model and output its JSON representation:

```bash
# Print JSON to standard output
onnx9000 json-extract my_model.onnx

# Write JSON to a file
onnx9000 json-extract my_model.onnx -o output.json
```

## Python SDK

In Python, you can utilize the core parser to load the graph and serialize it to JSON:

```python
import json
from onnx9000.core.parser.core import load

graph = load("my_model.onnx")

def custom_serializer(obj):
    if isinstance(obj, (bytes, bytearray)):
        return f"[Buffer: {len(obj)} bytes]"
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, set):
        return list(obj)
    return str(obj)

json_data = json.dumps(graph, default=custom_serializer, indent=2)
print(json_data)
```

## JavaScript/TypeScript SDK

In the browser or Node.js, the `load` method natively returns a JavaScript object that can be stringified:

```typescript
import { load } from '@onnx9000/core';
import * as fs from 'fs';

const arrayBuffer = fs.readFileSync('my_model.onnx').buffer;
const graph = await load(arrayBuffer);

const jsonString = JSON.stringify(
  graph,
  (key, value) => {
    if (key === 'data' && ArrayBuffer.isView(value)) {
      return `[Buffer: ${value.byteLength} bytes]`;
    }
    if (typeof value === 'bigint') {
      return value.toString() + 'n';
    }
    return value;
  },
  2,
);

console.log(jsonString);
```

## Interactive Web Demo

To try JSON extraction locally via our web interface, run:

```bash
onnx9000 serve
```

And navigate to `/json-extract` in your web browser.
