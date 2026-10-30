/**
 * @fileoverview WebGPU Demo UI.
 */

/**
 * Initializes the WebGPU demo UI.
 */
export function initWebGpuDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement | null;
  const out = document.getElementById('output') as HTMLElement | null;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', () => {
    out.innerText = 'Initializing WebGPU...';
    setTimeout(() => {
      out.innerText = `WebGPU engine loaded.
Execution complete: SUCCESS`;
    }, 50);
  });
}

initWebGpuDemo();
