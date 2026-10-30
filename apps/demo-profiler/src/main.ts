/**
 * @fileoverview Profiler Demo UI.
 */

/**
 * Initializes the profiler demo.
 */
export function initProfilerDemo(): void {
  document.getElementById('btn-run')?.addEventListener('click', () => {
    const output = document.getElementById('output');
    if (output) {
      output.textContent = `Initializing profiler...
`;
      setTimeout(() => {
        output.textContent += `[OK] Captured traces
`;
        output.textContent += '[OK] execution complete';
      }, 50);
    }
  });
}

initProfilerDemo();
