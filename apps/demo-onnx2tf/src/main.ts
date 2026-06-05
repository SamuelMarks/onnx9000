/**
 * Initializes the onnx2tf demo UI.
 */
export function initOnnx2TfDemo(): void {
  const convertBtn = document.getElementById('convertBtn') as HTMLButtonElement;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;
  const modelPathInput = document.getElementById('modelPath') as HTMLInputElement;
  const outputPathInput = document.getElementById('outputPath') as HTMLInputElement;
  const int8Quant = document.getElementById('int8Quant') as HTMLInputElement;

  if (!convertBtn || !resetBtn || !outputDiv) return;

  const log = (msg: string) => {
    outputDiv.textContent += msg + '\n';
  };

  convertBtn.addEventListener('click', () => {
    convertBtn.disabled = true;
    outputDiv.textContent = '';

    const model = modelPathInput?.value || '';
    const output = outputPathInput?.value || '';
    const isInt8 = int8Quant?.checked || false;

    log(`Loading ONNX model ${model}...`);

    setTimeout(() => {
      log(`Converting to TFLite format${isInt8 ? ' with INT8 quantization' : ''}...`);

      setTimeout(() => {
        log('Transpiling structural loops and applying fused activations...');
        log(`Saving TFLite model to ${output}...`);

        setTimeout(() => {
          log('onnx2tf conversion complete.');
          resetBtn.disabled = false;
        }, 500);
      }, 800);
    }, 500);
  });

  resetBtn.addEventListener('click', () => {
    outputDiv.textContent = 'Waiting to convert...\n';
    convertBtn.disabled = false;
    resetBtn.disabled = true;
  });
}

document.addEventListener('DOMContentLoaded', initOnnx2TfDemo);
if (document.readyState === 'complete' || document.readyState === 'interactive') {
  initOnnx2TfDemo();
}
