/**
 * @fileoverview ONNX2TF Converter Demo UI.
 */

/**
 * Initializes the onnx2tf demo UI.
 */
export function initOnnx2TfDemo(): void {
  const convertBtn = document.getElementById('convertBtn') as HTMLButtonElement | null;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement | null;
  const outputDiv = document.getElementById('output') as HTMLDivElement | null;
  const modelPathInput = document.getElementById('modelPath') as HTMLInputElement | null;
  const outputPathInput = document.getElementById('outputPath') as HTMLInputElement | null;
  const int8Quant = document.getElementById('int8Quant') as HTMLInputElement | null;

  if (!convertBtn || !resetBtn || !outputDiv) return;

  const log = (msg: string) => {
    outputDiv.textContent += `${msg}
`;
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
        }, 40);
      }, 40);
    }, 40);
  });

  resetBtn.addEventListener('click', () => {
    outputDiv.textContent = `Waiting to convert...
`;
    convertBtn.disabled = false;
    resetBtn.disabled = true;
  });
}

initOnnx2TfDemo();
