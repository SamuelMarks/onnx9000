/**
 * @fileoverview Triton Server Demo UI.
 */

/**
 * Initializes the Triton Server demo UI.
 */
export function initTritonServerDemo(): void {
  document.getElementById('btn-run')?.addEventListener('click', () => {
    const output = document.getElementById('output');
    if (output) {
      output.textContent = `Running...
`;
      setTimeout(() => {
        output.textContent += '[OK] Triton Server execution complete.';
      }, 50);
    }
  });
}

initTritonServerDemo();
