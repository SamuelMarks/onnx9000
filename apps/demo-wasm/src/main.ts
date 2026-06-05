/**
 * Initializes the WASM demo UI.
 */
export function initWasmDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
  const out = document.getElementById('output') as HTMLElement;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', () => {
    out.innerText = 'Initializing WASM Compiler...';
    setTimeout(() => {
      out.innerText = 'WASM engine loaded.\nExecution complete: SUCCESS';
    }, 500);
  });
}
initWasmDemo();
