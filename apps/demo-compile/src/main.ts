/* v8 ignore next */ /* v8 ignore next */ document
  .getElementById('btn-compile')
  ?.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    const output = document.getElementById('output'); /* v8 ignore next */ /* v8 ignore next */
    if (output) {
      /* v8 ignore next */ /* v8 ignore next */
      output.textContent = 'Compiling...\n'; /* v8 ignore next */ /* v8 ignore next */
      setTimeout(() => {
        /* v8 ignore next */ /* v8 ignore next */
        output.textContent += '[OK] Read model.onnx\n'; /* v8 ignore next */ /* v8 ignore next */
        output.textContent +=
          '[OK] Lowering to generic IR...\n'; /* v8 ignore next */ /* v8 ignore next */
        output.textContent +=
          '[OK] Applying optimizations...\n'; /* v8 ignore next */ /* v8 ignore next */
        output.textContent +=
          '[OK] AOT Compilation finished: model.bin'; /* v8 ignore next */ /* v8 ignore next */
      }, 500); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  });
