/**
 * @fileoverview Progressive Model Loading Demo UI.
 */
import { loadProgressive, type ProgressiveSession } from '@onnx9000/backend-web';

/**
 * Initializes the progressive loading demo UI.
 */
export function initProgressiveLoadingDemo(): void {
  const modelUrlInput = document.getElementById('modelUrl') as HTMLInputElement | null;
  const loadBtn = document.getElementById('loadBtn') as HTMLButtonElement | null;
  const runBtn = document.getElementById('runBtn') as HTMLButtonElement | null;
  const outputDiv = document.getElementById('output') as HTMLDivElement | null;

  if (!modelUrlInput || !loadBtn || !runBtn || !outputDiv) return;

  let session: ProgressiveSession | null = null;

  loadBtn.addEventListener('click', async () => {
    outputDiv.textContent = 'Initializing progressive session...';
    try {
      session = await loadProgressive(modelUrlInput.value, {
        maxChunkSize: 1024 * 1024,
      });
      outputDiv.textContent = `Session initialized! The model is not loaded yet.
Click "Run Inference" to trigger chunked downloading.`;
      runBtn.disabled = false;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      outputDiv.textContent = `Error: ${msg}`;
    }
  });

  runBtn.addEventListener('click', async () => {
    if (!session) return;
    outputDiv.textContent = `Running inference...
Starting progressive tensor streaming...`;
    try {
      const outputs = await session.run({});
      outputDiv.textContent += `

Success! Progressively loaded weights and completed inference.
Output: ${JSON.stringify(outputs, null, 2)}`;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      outputDiv.textContent += `

Error during inference: ${msg}`;
    }
  });
}

initProgressiveLoadingDemo();
