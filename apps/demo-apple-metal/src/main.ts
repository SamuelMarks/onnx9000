/**
 * @fileoverview Apple Metal Demo UI.
 */

/**
 * Initializes the Apple Metal demo UI.
 */
export function initAppleMetalDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement | null;
  const out = document.getElementById('output') as HTMLElement | null;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', () => {
    out.innerText = 'Initializing Apple Metal...';
    setTimeout(() => {
      out.innerText = `Apple Metal engine loaded.
Execution complete: SUCCESS`;
    }, 50);
  });
}

initAppleMetalDemo();
