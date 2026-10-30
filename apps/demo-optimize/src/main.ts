/**
 * @fileoverview Optimize Graph Demo UI.
 */

/**
 * Initializes the optimize demo UI.
 */
export function initOptimizeDemo(): void {
  const optimizeBtn = document.getElementById('optimizeBtn') as HTMLButtonElement | null;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement | null;
  const outputDiv = document.getElementById('output') as HTMLDivElement | null;
  const modelPathInput = document.getElementById('modelPath') as HTMLInputElement | null;
  const outputPathInput = document.getElementById('outputPath') as HTMLInputElement | null;
  const passesInput = document.getElementById('passes') as HTMLInputElement | null;

  if (!optimizeBtn || !resetBtn || !outputDiv) return;

  const log = (msg: string) => {
    outputDiv.textContent += `${msg}
`;
  };

  optimizeBtn.addEventListener('click', () => {
    optimizeBtn.disabled = true;
    outputDiv.textContent = '';

    const model = modelPathInput?.value || '';
    const output = outputPathInput?.value || '';
    const passes = passesInput?.value || 'default';

    log(`Loading ONNX model ${model}...`);

    setTimeout(() => {
      log(`Running optimization passes: ${passes}`);

      setTimeout(() => {
        log(' - Identified 4 nodes for fusion');
        log(' - Eliminated 2 deadends');
        log(`Saving optimized model to ${output}...`);

        setTimeout(() => {
          log('Graph optimization complete.');
          resetBtn.disabled = false;
        }, 40);
      }, 40);
    }, 40);
  });

  resetBtn.addEventListener('click', () => {
    outputDiv.textContent = `Waiting to optimize...
`;
    optimizeBtn.disabled = false;
    resetBtn.disabled = true;
  });
}

initOptimizeDemo();
