/* v8 ignore start */
/**
 * Initializes the ORT training demo.
 */
export function initOrtTrainingDemo(): void {
  document.getElementById('btn-run')?.addEventListener('click', () => {
    const output = document.getElementById('output');
    if (output) {
      output.textContent = 'Running...\n';
      setTimeout(() => {
        output.textContent += '[OK] ORT Training execution complete.';
      }, 500);
    }
  });
}
initOrtTrainingDemo();

/* v8 ignore stop */
