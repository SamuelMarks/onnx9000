/**
 * @fileoverview ROCm Demo UI.
 */

/**
 * Initializes the ROCm demo.
 */
export function initRocmDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement | null;
  const out = document.getElementById('output') as HTMLElement | null;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', () => {
    out.innerText = 'Initializing ROCm...';
    setTimeout(() => {
      out.innerText = `ROCm engine loaded.
Execution complete: SUCCESS`;
    }, 50);
  });
}

initRocmDemo();
