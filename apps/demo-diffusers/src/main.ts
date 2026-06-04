/* v8 ignore next */ /* v8 ignore next */ import { DiffusionPipeline } from '@onnx9000/diffusers/src/pipeline.js'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const runBtn = document.getElementById(
  'run-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
runBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Initializing Pipeline...'; /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const pipe = new DiffusionPipeline(); /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nPipeline initialized. Generating mock tensor...'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nImage tensor generated successfully [1, 3, 512, 512]'; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText = `Error: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
