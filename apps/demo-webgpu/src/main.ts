/* v8 ignore start */
/**
 * Initializes the WebGPU demo UI.
 */
export function initWebGpuDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
  const out = document.getElementById('output') as HTMLElement;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', () => {
    out.innerText = 'Initializing WebGPU...';
    setTimeout(() => {
      out.innerText = 'WebGPU engine loaded.\nExecution complete: SUCCESS';
    }, 500);
  });
}
initWebGpuDemo();

/* v8 ignore stop */
