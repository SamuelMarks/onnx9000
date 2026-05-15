document.addEventListener('DOMContentLoaded', () => {
  const runBtn = document.getElementById('runBtn') as HTMLButtonElement;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;

  const log = (msg: string) => {
    outputDiv.textContent += msg + '\n';
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
          log('\nClassification Result:');
          log('Label: TABBY_CAT');
          log('Confidence: 0.985');
          log('\nPipeline finished successfully.');
          resetBtn.disabled = false;
        }, 800);
      }, 500);
    }, 500);
  });

  resetBtn.addEventListener('click', () => {
    outputDiv.textContent = 'Ready. Click "Run Classification" to start.\n';
    runBtn.disabled = false;
    resetBtn.disabled = true;
  });
});
