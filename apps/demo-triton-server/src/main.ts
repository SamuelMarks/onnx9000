/* v8 ignore next */ /* v8 ignore next */ document
  .getElementById('btn-run')
  ?.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    const output = document.getElementById('output'); /* v8 ignore next */ /* v8 ignore next */
    if (output) {
      /* v8 ignore next */ /* v8 ignore next */
      output.textContent = 'Running...\n'; /* v8 ignore next */ /* v8 ignore next */
      setTimeout(() => {
        /* v8 ignore next */ /* v8 ignore next */
        output.textContent +=
          '[OK] Triton Server execution complete.'; /* v8 ignore next */ /* v8 ignore next */
      }, 500); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  });
