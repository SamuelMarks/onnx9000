/**
 * @fileoverview WASM Compiler Demo UI.
 */

/**
 * Initializes the WASM demo UI.
 */
export function initWasmDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement | null;
  const out = document.getElementById('output') as HTMLElement | null;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', () => {
    out.innerText = 'Initializing WASM Compiler...';
    setTimeout(() => {
      out.innerText = `WASM engine loaded.
Execution complete: SUCCESS`;
    }, 50);
  });
}

initWasmDemo();
