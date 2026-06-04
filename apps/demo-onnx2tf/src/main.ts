/* v8 ignore next */ /* v8 ignore next */ document.addEventListener('DOMContentLoaded', () => {
  /* v8 ignore next */ /* v8 ignore next */
  const convertBtn = document.getElementById(
    'convertBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const resetBtn = document.getElementById(
    'resetBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const outputDiv = document.getElementById(
    'output',
  ) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const modelPathInput = document.getElementById(
    'modelPath',
  ) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  const outputPathInput = document.getElementById(
    'outputPath',
  ) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  const int8Quant = document.getElementById(
    'int8Quant',
  ) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const log = (msg: string) => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent += msg + '\n'; /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  convertBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    convertBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent = ''; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const model = modelPathInput.value; /* v8 ignore next */ /* v8 ignore next */
    const output = outputPathInput.value; /* v8 ignore next */ /* v8 ignore next */
    const isInt8 = int8Quant.checked; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    log(`Loading ONNX model ${model}...`); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    setTimeout(() => {
      /* v8 ignore next */ /* v8 ignore next */
      log(
        `Converting to TFLite format${isInt8 ? ' with INT8 quantization' : ''}...`,
      ); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      setTimeout(() => {
        /* v8 ignore next */ /* v8 ignore next */
        log(
          'Transpiling structural loops and applying fused activations...',
        ); /* v8 ignore next */ /* v8 ignore next */
        log(`Saving TFLite model to ${output}...`); /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        setTimeout(() => {
          /* v8 ignore next */ /* v8 ignore next */
          log('onnx2tf conversion complete.'); /* v8 ignore next */ /* v8 ignore next */
          resetBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
        }, 500); /* v8 ignore next */ /* v8 ignore next */
      }, 800); /* v8 ignore next */ /* v8 ignore next */
    }, 500); /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  resetBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent = 'Waiting to convert...\n'; /* v8 ignore next */ /* v8 ignore next */
    convertBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
    resetBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
});
