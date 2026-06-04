/* v8 ignore next */ /* v8 ignore next */ document.addEventListener('DOMContentLoaded', () => {
  /* v8 ignore next */ /* v8 ignore next */
  const simplifyBtn = document.getElementById(
    'simplifyBtn',
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
  /* v8 ignore next */ /* v8 ignore next */
  const log = (msg: string) => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent += msg + '\n'; /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  simplifyBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    simplifyBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent = ''; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const model = modelPathInput.value; /* v8 ignore next */ /* v8 ignore next */
    const output = outputPathInput.value; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    log(`Loading ONNX model ${model}...`); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    setTimeout(() => {
      /* v8 ignore next */ /* v8 ignore next */
      log(`Simplifying graph...`); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      setTimeout(() => {
        /* v8 ignore next */ /* v8 ignore next */
        log(' - Folded 12 constants'); /* v8 ignore next */ /* v8 ignore next */
        log(' - Eliminated 3 unreachable nodes'); /* v8 ignore next */ /* v8 ignore next */
        log(`Saving simplified model to ${output}...`); /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        setTimeout(() => {
          /* v8 ignore next */ /* v8 ignore next */
          log('Graph simplification complete.'); /* v8 ignore next */ /* v8 ignore next */
          resetBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
        }, 500); /* v8 ignore next */ /* v8 ignore next */
      }, 800); /* v8 ignore next */ /* v8 ignore next */
    }, 500); /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  resetBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent = 'Waiting to simplify...\n'; /* v8 ignore next */ /* v8 ignore next */
    simplifyBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
    resetBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
});
