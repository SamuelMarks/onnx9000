/**
 * JavaScript wrapper to invoke the onnx9000 pure-Python exporter 
 * from the browser via Pyodide.
 */

class ONNX9000Exporter {
    constructor(pyodide) {
        this.pyodide = pyodide;
    }

    /**
     * Initializes the Python environment with required packages.
     */
    async init() {
        await this.pyodide.loadPackage("numpy");
        // Assuming the wheel is built and available at this URL
        await this.pyodide.loadPackage("./onnx9000-0.1.0-py3-none-any.whl");
        
        this.pyodide.runPython(`
            import js
            import io
            from onnx9000.frontend.frontend import export
            from onnx9000.frontend.frontend.models import ResNet18, MobileNetV2, GPT2
            from onnx9000.frontend.frontend.tensor import Tensor
            from onnx9000.core.dtypes import DType
        `);
    }

    /**
     * Traces and exports a model, returning the ONNX binary as a Uint8Array.
     */
    async exportModel(modelName, shape, dtypeStr) {
        const result = this.pyodide.runPython(`
            def run_export():
                model = None
                if "${modelName}" == "ResNet18":
                    model = ResNet18()
                elif "${modelName}" == "MobileNetV2":
                    model = MobileNetV2()
                elif "${modelName}" == "GPT2":
                    model = GPT2()
                else:
                    raise ValueError(f"Unknown model {modelName}")
                
                dt = getattr(DType, "${dtypeStr}".upper())
                x = Tensor(${JSON.stringify(shape)}, dt, "input")
                
                buffer = io.BytesIO()
                export(model, x, buffer)
                return buffer.getvalue()
                
            run_export()
        `);
        
        return new Uint8Array(result.toJs());
    }
}

if (typeof module !== 'undefined') {
    module.exports = { ONNX9000Exporter };
}
