/**
 * @fileoverview Native CPU Demo UI.
 */

/**
 * Initializes the Native CPU demo.
 */
export function initNativeCpuDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement | null;
  const out = document.getElementById('output') as HTMLElement | null;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', () => {
    out.innerText = 'Initializing Native CPU...';
    setTimeout(() => {
      out.innerText = `Native CPU engine loaded.
Execution complete: SUCCESS`;
    }, 50);
  });
}

initNativeCpuDemo();
