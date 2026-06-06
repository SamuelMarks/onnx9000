/* v8 ignore start */
import { loadProgressive, type ProgressiveSession } from '@onnx9000/backend-web';

/**
 * Initializes the progressive loading demo UI.
 */
export function initProgressiveLoadingDemo(): void {
  const modelUrlInput = document.getElementById('modelUrl') as HTMLInputElement;
  const loadBtn = document.getElementById('loadBtn') as HTMLButtonElement;
  const runBtn = document.getElementById('runBtn') as HTMLButtonElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;

  if (!modelUrlInput || !loadBtn || !runBtn || !outputDiv) return;

  let session: ProgressiveSession | null = null;

  loadBtn.addEventListener('click', async () => {
    outputDiv.textContent = 'Initializing progressive session...';
    try {
      session = await loadProgressive(modelUrlInput.value, {
        maxChunkSize: 1024 * 1024,
      });
      outputDiv.textContent =
        'Session initialized! The model is not loaded yet.\nClick "Run Inference" to trigger chunked downloading.';
      runBtn.disabled = false;
    } catch (err: any) {
      outputDiv.textContent = `Error: ${err.message}`;
    }
  });

  runBtn.addEventListener('click', async () => {
    if (!session) return;
    outputDiv.textContent = 'Running inference...\nStarting progressive tensor streaming...';
    try {
      const outputs = await session.run({});
      outputDiv.textContent +=
        '\n\nSuccess! Progressively loaded weights and completed inference.\nOutput: ' +
        JSON.stringify(outputs, null, 2);
    } catch (err: any) {
      outputDiv.textContent += `\n\nError during inference: ${err.message}`;
    }
  });
}

document.addEventListener('DOMContentLoaded', initProgressiveLoadingDemo);
if (document.readyState === 'complete' || document.readyState === 'interactive') {
  initProgressiveLoadingDemo();
}

/* v8 ignore stop */
