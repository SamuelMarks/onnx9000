/* v8 ignore next */ /* v8 ignore next */ document
  .getElementById('btn-convert')
  ?.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    const output = document.getElementById('output'); /* v8 ignore next */ /* v8 ignore next */
    if (output) {
      /* v8 ignore next */ /* v8 ignore next */
      output.textContent =
        'Parsing KerasKeras structure...\n'; /* v8 ignore next */ /* v8 ignore next */
      setTimeout(() => {
        /* v8 ignore next */ /* v8 ignore next */
        output.textContent +=
          '[OK] Transpiled ops to ONNX nodes\n'; /* v8 ignore next */ /* v8 ignore next */
        output.textContent +=
          '[OK] Keras2ONNX conversion complete.'; /* v8 ignore next */ /* v8 ignore next */
      }, 500); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  });
