/**
 * @fileoverview Simplify Graph Demo UI.
 */

/**
 * Initializes the simplify demo UI.
 */
export function initSimplifyDemo(): void {
  const simplifyBtn = document.getElementById('simplifyBtn') as HTMLButtonElement | null;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement | null;
  const outputDiv = document.getElementById('output') as HTMLDivElement | null;
  const modelPathInput = document.getElementById('modelPath') as HTMLInputElement | null;
  const outputPathInput = document.getElementById('outputPath') as HTMLInputElement | null;

  if (!simplifyBtn || !resetBtn || !outputDiv) return;

  const log = (msg: string) => {
    outputDiv.textContent += `${msg}
`;
  };

  simplifyBtn.addEventListener('click', () => {
    simplifyBtn.disabled = true;
    outputDiv.textContent = '';

    const model = modelPathInput?.value || '';
    const output = outputPathInput?.value || '';

    log(`Loading ONNX model ${model}...`);

    setTimeout(() => {
      log('Simplifying graph...');

      setTimeout(() => {
        log(' - Folded 12 constants');
        log(' - Eliminated 3 unreachable nodes');
        log(`Saving simplified model to ${output}...`);

        setTimeout(() => {
          log('Graph simplification complete.');
          resetBtn.disabled = false;
        }, 40);
      }, 40);
    }, 40);
  });

  resetBtn.addEventListener('click', () => {
    outputDiv.textContent = `Waiting to simplify...
`;
    simplifyBtn.disabled = false;
    resetBtn.disabled = true;
  });
}

initSimplifyDemo();
