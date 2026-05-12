import { pipeline } from '@onnx9000/transformers';

const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const out = document.getElementById('transformers-output') as HTMLElement;

runBtn.addEventListener('click', async () => {
  out.innerText = 'Initializing Pipeline...\n';
  runBtn.disabled = true;

  try {
    const pipe = await pipeline('text-classification');
    out.innerText += '\nPipeline initialized for text-classification.';

    out.innerText += '\nRunning inference on "I love ONNX9000!"...';

    // We expect postprocess to return { label: 'positive', score: 0.99... } based on the mock in the package
    const result = await pipe('I love ONNX9000!');

    out.innerText += `\n\nResult: ${JSON.stringify(result)}`;
    out.innerText += '\n\nSuccess! Transformers.js pipeline ran successfully.';
  } catch (e: any) {
    out.innerText += `\nError: ${e.message}`;
  } finally {
    runBtn.disabled = false;
  }
});
