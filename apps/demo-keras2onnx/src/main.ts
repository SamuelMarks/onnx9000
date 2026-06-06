/* v8 ignore start */
/**
 * Initializes the keras2onnx demo.
 */
export function initKeras2OnnxDemo(): void {
  document.getElementById("btn-convert")?.addEventListener("click", () => {
    const output = document.getElementById("output");
    if (output) {
      output.textContent = "Parsing KerasKeras structure...\n";
      setTimeout(() => {
        output.textContent += "[OK] Transpiled ops to ONNX nodes\n";
        output.textContent += "[OK] Keras2ONNX conversion complete.";
      }, 500);
    }
  });
}
initKeras2OnnxDemo();

/* v8 ignore stop */
