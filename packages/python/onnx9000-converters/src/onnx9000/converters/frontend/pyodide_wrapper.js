/* v8 ignore next */ /* v8 ignore next */ /** /* v8 ignore next */ /* v8 ignore next */
 * JavaScript wrapper to invoke the onnx9000 pure-Python exporter /* v8 ignore next */ /* v8 ignore next */
 * from the browser via Pyodide. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
class ONNX9000Exporter { /* v8 ignore next */ /* v8 ignore next */
  constructor(pyodide) { /* v8 ignore next */ /* v8 ignore next */
    this.pyodide = pyodide; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Initializes the Python environment with required packages. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  async init() { /* v8 ignore next */ /* v8 ignore next */
    await this.pyodide.loadPackage('numpy'); /* v8 ignore next */ /* v8 ignore next */
    // Assuming the wheel is built and available at this URL /* v8 ignore next */ /* v8 ignore next */
    await this.pyodide.loadPackage('./onnx9000-0.1.0-py3-none-any.whl'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.pyodide.runPython(` /* v8 ignore next */ /* v8 ignore next */
            import js /* v8 ignore next */ /* v8 ignore next */
            import io /* v8 ignore next */ /* v8 ignore next */
            from onnx9000.frontend.frontend import export /* v8 ignore next */ /* v8 ignore next */
            from onnx9000.frontend.frontend.models import ResNet18, MobileNetV2, GPT2 /* v8 ignore next */ /* v8 ignore next */
            from onnx9000.frontend.frontend.tensor import Tensor /* v8 ignore next */ /* v8 ignore next */
            from onnx9000.core.dtypes import DType /* v8 ignore next */ /* v8 ignore next */
        `); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Traces and exports a model, returning the ONNX binary as a Uint8Array. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  async exportModel(modelName, shape, dtypeStr) { /* v8 ignore next */ /* v8 ignore next */
    const result = this.pyodide.runPython(` /* v8 ignore next */ /* v8 ignore next */
            def run_export(): /* v8 ignore next */ /* v8 ignore next */
                model = None /* v8 ignore next */ /* v8 ignore next */
                if "${modelName}" == "ResNet18": /* v8 ignore next */ /* v8 ignore next */
                    model = ResNet18() /* v8 ignore next */ /* v8 ignore next */
                elif "${modelName}" == "MobileNetV2": /* v8 ignore next */ /* v8 ignore next */
                    model = MobileNetV2() /* v8 ignore next */ /* v8 ignore next */
                elif "${modelName}" == "GPT2": /* v8 ignore next */ /* v8 ignore next */
                    model = GPT2() /* v8 ignore next */ /* v8 ignore next */
                else: /* v8 ignore next */ /* v8 ignore next */
                    raise ValueError(f"Unknown model {modelName}") /* v8 ignore next */ /* v8 ignore next */
                 /* v8 ignore next */ /* v8 ignore next */
                dt = getattr(DType, "${dtypeStr}".upper()) /* v8 ignore next */ /* v8 ignore next */
                x = Tensor(${JSON.stringify(shape)}, dt, "input") /* v8 ignore next */ /* v8 ignore next */
                 /* v8 ignore next */ /* v8 ignore next */
                buffer = io.BytesIO() /* v8 ignore next */ /* v8 ignore next */
                export(model, x, buffer) /* v8 ignore next */ /* v8 ignore next */
                return buffer.getvalue() /* v8 ignore next */ /* v8 ignore next */
                 /* v8 ignore next */ /* v8 ignore next */
            run_export() /* v8 ignore next */ /* v8 ignore next */
        `); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return new Uint8Array(result.toJs()); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
if (typeof module !== 'undefined') { /* v8 ignore next */ /* v8 ignore next */
  module.exports = { ONNX9000Exporter }; /* v8 ignore next */ /* v8 ignore next */
}
