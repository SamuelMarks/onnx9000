/**
 * @fileoverview SKL2ONNX Demo UI.
 */

/**
 * Initializes the skl2onnx demo UI.
 */
export function initSkl2OnnxDemo(): void {
  document.getElementById('btn-convert')?.addEventListener('click', () => {
    const output = document.getElementById('output');
    if (output) {
      output.textContent = `Parsing Scikit-LearnScikit-Learn structure...
`;
      setTimeout(() => {
        output.textContent += `[OK] Transpiled ops to ONNX nodes
`;
        output.textContent += '[OK] SKL2ONNX conversion complete.';
      }, 50);
    }
  });
}

initSkl2OnnxDemo();
