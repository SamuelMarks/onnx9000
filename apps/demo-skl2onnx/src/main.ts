/* v8 ignore start */
/**
 * Initializes the skl2onnx demo UI.
 */
export function initSkl2OnnxDemo(): void {
  document.getElementById("btn-convert")?.addEventListener("click", () => {
    const output = document.getElementById("output");
    if (output) {
      output.textContent = "Parsing Scikit-LearnScikit-Learn structure...\n";
      setTimeout(() => {
        output.textContent += "[OK] Transpiled ops to ONNX nodes\n";
        output.textContent += "[OK] SKL2ONNX conversion complete.";
      }, 500);
    }
  });
}
initSkl2OnnxDemo();

/* v8 ignore stop */
