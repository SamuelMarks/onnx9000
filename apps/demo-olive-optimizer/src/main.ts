/**
 * @fileoverview Olive Optimizer Demo UI.
 */

/**
 * Initializes the Olive optimizer demo UI.
 */
export function initOliveOptimizerDemo(): void {
  document.getElementById('btn-run')?.addEventListener('click', () => {
    const output = document.getElementById('output');
    if (output) {
      output.textContent = `Running...
`;
      setTimeout(() => {
        output.textContent += '[OK] Olive Optimizer execution complete.';
      }, 50);
    }
  });
}

initOliveOptimizerDemo();
