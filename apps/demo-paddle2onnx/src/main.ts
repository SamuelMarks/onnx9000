/**
 * @fileoverview Paddle2ONNX Demo UI.
 */

/**
 * Initializes the paddle2onnx demo.
 */
export function initPaddle2OnnxDemo(): void {
  document.getElementById('btn-convert')?.addEventListener('click', () => {
    const output = document.getElementById('output');
    if (output) {
      output.textContent = `Parsing PaddlePaddle structure...
`;
      setTimeout(() => {
        output.textContent += `[OK] Transpiled ops to ONNX nodes
`;
        output.textContent += '[OK] Paddle2ONNX conversion complete.';
      }, 50);
    }
  });
}

initPaddle2OnnxDemo();
