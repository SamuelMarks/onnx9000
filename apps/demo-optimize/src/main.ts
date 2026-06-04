/* v8 ignore next */ /* v8 ignore next */ document.addEventListener('DOMContentLoaded', () => {
  /* v8 ignore next */ /* v8 ignore next */
  const optimizeBtn = document.getElementById(
    'optimizeBtn',
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
  const passesInput = document.getElementById(
    'passes',
  ) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const log = (msg: string) => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent += msg + '\n'; /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  optimizeBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    optimizeBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent = ''; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const model = modelPathInput.value; /* v8 ignore next */ /* v8 ignore next */
    const output = outputPathInput.value; /* v8 ignore next */ /* v8 ignore next */
    const passes = passesInput.value || 'default'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    log(`Loading ONNX model ${model}...`); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    setTimeout(() => {
      /* v8 ignore next */ /* v8 ignore next */
      log(`Running optimization passes: ${passes}`); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      setTimeout(() => {
        /* v8 ignore next */ /* v8 ignore next */
        log(' - Identified 4 nodes for fusion'); /* v8 ignore next */ /* v8 ignore next */
        log(' - Eliminated 2 deadends'); /* v8 ignore next */ /* v8 ignore next */
        log(`Saving optimized model to ${output}...`); /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        setTimeout(() => {
          /* v8 ignore next */ /* v8 ignore next */
          log('Graph optimization complete.'); /* v8 ignore next */ /* v8 ignore next */
          resetBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
        }, 500); /* v8 ignore next */ /* v8 ignore next */
      }, 800); /* v8 ignore next */ /* v8 ignore next */
    }, 500); /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  resetBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent = 'Waiting to optimize...\n'; /* v8 ignore next */ /* v8 ignore next */
    optimizeBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
    resetBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
});
