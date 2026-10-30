/**
 * @fileoverview Keras2ONNX Demo UI.
 */

/**
 * Initializes the keras2onnx demo.
 */
export function initKeras2OnnxDemo(): void {
  document.getElementById('btn-convert')?.addEventListener('click', () => {
    const output = document.getElementById('output');
    if (output) {
      output.textContent = `Parsing KerasKeras structure...
`;
      setTimeout(() => {
        output.textContent += `[OK] Transpiled ops to ONNX nodes
`;
        output.textContent += '[OK] Keras2ONNX conversion complete.';
      }, 50);
    }
  });
}

initKeras2OnnxDemo();
