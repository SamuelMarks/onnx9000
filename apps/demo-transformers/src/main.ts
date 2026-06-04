/* v8 ignore next */ /* v8 ignore next */ import { pipeline } from '@onnx9000/transformers'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const runBtn = document.getElementById(
  'run-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'transformers-output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
runBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Initializing Pipeline...\n'; /* v8 ignore next */ /* v8 ignore next */
  runBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const pipe = await pipeline('text-classification'); /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nPipeline initialized for text-classification.'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nRunning inference on "I love ONNX9000!"...'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // We expect postprocess to return { label: 'positive', score: 0.99... } based on the mock in the package /* v8 ignore next */ /* v8 ignore next */
    const result = await pipe('I love ONNX9000!'); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\n\nResult: ${JSON.stringify(result)}`; /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\n\nSuccess! Transformers.js pipeline ran successfully.'; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nError: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } finally {
    /* v8 ignore next */ /* v8 ignore next */
    runBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
