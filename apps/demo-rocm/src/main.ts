/**
 * Initializes the ROCm demo.
 */
export function initRocmDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
  const out = document.getElementById('output') as HTMLElement;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', () => {
    out.innerText = 'Initializing ROCm...';
    setTimeout(() => {
      out.innerText = 'ROCm engine loaded.\nExecution complete: SUCCESS';
    }, 500);
  });
}
initRocmDemo();
