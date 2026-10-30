/**
 * @fileoverview Zero-Dependency Image Classifier Demo UI.
 */

/**
 * Initializes the zero-dependency classifier demo UI.
 */
export function initZeroDepClassifierDemo(): void {
  const runBtn = document.getElementById('runBtn') as HTMLButtonElement | null;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement | null;
  const outputDiv = document.getElementById('output') as HTMLDivElement | null;

  if (!runBtn || !resetBtn || !outputDiv) return;

  const log = (msg: string) => {
    outputDiv.textContent += `${msg}
`;
  };

  runBtn.addEventListener('click', () => {
    runBtn.disabled = true;
    outputDiv.textContent = '';
    log('Initializing zero-dependency classification pipeline...');
    log('Loading small tokenizer config from static assets...');

    setTimeout(() => {
      log('Building inference graph natively in JS...');

      setTimeout(() => {
        log('Classifying input image (mock buffer): Float32Array(224 * 224 * 3)');
        log('Running operations natively (WebGPU fallback -> WASM)...');

        setTimeout(() => {
          log(`
Classification Result:`);
          log('Label: TABBY_CAT');
          log('Confidence: 0.985');
          log(`
Pipeline finished successfully.`);
          resetBtn.disabled = false;
        }, 40);
      }, 40);
    }, 40);
  });

  resetBtn.addEventListener('click', () => {
    outputDiv.textContent = `Ready. Click "Run Classification" to start.
`;
    runBtn.disabled = false;
    resetBtn.disabled = true;
  });
}

initZeroDepClassifierDemo();
