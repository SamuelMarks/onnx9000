document.addEventListener('DOMContentLoaded', () => {
  const optimizeBtn = document.getElementById('optimizeBtn') as HTMLButtonElement;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;

  const modelPathInput = document.getElementById('modelPath') as HTMLInputElement;
  const outputPathInput = document.getElementById('outputPath') as HTMLInputElement;
  const passesInput = document.getElementById('passes') as HTMLInputElement;

  const log = (msg: string) => {
    outputDiv.textContent += msg + '\n';
  };

  optimizeBtn.addEventListener('click', () => {
    optimizeBtn.disabled = true;
    outputDiv.textContent = '';

    const model = modelPathInput.value;
    const output = outputPathInput.value;
    const passes = passesInput.value || 'default';

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
        }, 500);
      }, 800);
    }, 500);
  });

  resetBtn.addEventListener('click', () => {
    outputDiv.textContent = 'Waiting to optimize...\n';
    optimizeBtn.disabled = false;
    resetBtn.disabled = true;
  });
});
