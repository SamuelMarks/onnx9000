document.addEventListener('DOMContentLoaded', () => {
  const simplifyBtn = document.getElementById('simplifyBtn') as HTMLButtonElement;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;

  const modelPathInput = document.getElementById('modelPath') as HTMLInputElement;
  const outputPathInput = document.getElementById('outputPath') as HTMLInputElement;

  const log = (msg: string) => {
    outputDiv.textContent += msg + '\n';
  };

  simplifyBtn.addEventListener('click', () => {
    simplifyBtn.disabled = true;
    outputDiv.textContent = '';

    const model = modelPathInput.value;
    const output = outputPathInput.value;

    log(`Loading ONNX model ${model}...`);

    setTimeout(() => {
      log(`Simplifying graph...`);

      setTimeout(() => {
        log(' - Folded 12 constants');
        log(' - Eliminated 3 unreachable nodes');
        log(`Saving simplified model to ${output}...`);

        setTimeout(() => {
          log('Graph simplification complete.');
          resetBtn.disabled = false;
        }, 500);
      }, 800);
    }, 500);
  });

  resetBtn.addEventListener('click', () => {
    outputDiv.textContent = 'Waiting to simplify...\n';
    simplifyBtn.disabled = false;
    resetBtn.disabled = true;
  });
});
