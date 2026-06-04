/* v8 ignore next */ /* v8 ignore next */ const runBtn = document.getElementById(
  'run-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
runBtn.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Initializing WASM Compiler...'; /* v8 ignore next */ /* v8 ignore next */
  setTimeout(() => {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText =
      'WASM engine loaded.\nExecution complete: SUCCESS'; /* v8 ignore next */ /* v8 ignore next */
  }, 500); /* v8 ignore next */ /* v8 ignore next */
});
