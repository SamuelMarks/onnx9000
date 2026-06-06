/* v8 ignore start */
/**
 * Initializes the Triton Server demo UI.
 */
export function initTritonServerDemo(): void {
  document.getElementById('btn-run')?.addEventListener('click', () => {
    const output = document.getElementById('output');
    if (output) {
      output.textContent = 'Running...\n';
      setTimeout(() => {
        output.textContent += '[OK] Triton Server execution complete.';
      }, 500);
    }
  });
}
initTritonServerDemo();

/* v8 ignore stop */
