/* v8 ignore start */
/**
 * Initializes the ONNX tool demo.
 */
export function initOnnxToolDemo(): void {
  document.getElementById('btn-run')?.addEventListener('click', () => {
    const output = document.getElementById('output');
    if (output) {
      output.textContent = 'Running...\n';
      setTimeout(() => {
        output.textContent += '[OK] ONNX Tool execution complete.';
      }, 500);
    }
  });
}
initOnnxToolDemo();

/* v8 ignore stop */
