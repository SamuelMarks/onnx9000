/**
 * @fileoverview Transformers.js Pipeline Demo UI.
 */
import { pipeline } from '@onnx9000/transformers';

/**
 * Initializes the Transformers demo UI.
 */
export function initTransformersDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement | null;
  const out = document.getElementById('transformers-output') as HTMLElement | null;
  if (!runBtn || !out) return;

  runBtn.addEventListener('click', async () => {
    out.innerText = 'Initializing Transformers Pipeline (text-classification)...';
    runBtn.disabled = true;

    try {
      const pipe = await pipeline('text-classification');
      out.innerText += ' Pipeline initialized.';
      out.innerText += ' Running inference on: "I love ONNX9000!"...';

      const result = await pipe('I love ONNX9000!');

      out.innerText += ` Result: ${JSON.stringify(result)}`;
      out.innerText += ' Success! Transformers.js pipeline ran successfully.';
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      out.innerText += ` Error: ${msg}`;
    } finally {
      runBtn.disabled = false;
    }
  });
}

initTransformersDemo();
