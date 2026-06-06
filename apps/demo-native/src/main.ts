/* v8 ignore start */
/**
 * Initializes the Native CPU demo.
 */
export function initNativeCpuDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
  const out = document.getElementById('output') as HTMLElement;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', () => {
    out.innerText = 'Initializing Native CPU...';
    setTimeout(() => {
      out.innerText = 'Native CPU engine loaded.\nExecution complete: SUCCESS';
    }, 500);
  });
}
initNativeCpuDemo();

/* v8 ignore stop */
