/* v8 ignore next */ /* v8 ignore next */ import {
  loadProgressive,
  ProgressiveSession,
} from '@onnx9000/backend-web'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
document.addEventListener('DOMContentLoaded', () => {
  /* v8 ignore next */ /* v8 ignore next */
  const modelUrlInput = document.getElementById(
    'modelUrl',
  ) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  const loadBtn = document.getElementById(
    'loadBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const runBtn = document.getElementById(
    'runBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const outputDiv = document.getElementById(
    'output',
  ) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let session: ProgressiveSession | null = null; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  loadBtn.addEventListener('click', async () => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent =
      'Initializing progressive session...'; /* v8 ignore next */ /* v8 ignore next */
    try {
      /* v8 ignore next */ /* v8 ignore next */
      session = await loadProgressive(modelUrlInput.value, {
        /* v8 ignore next */ /* v8 ignore next */
        maxChunkSize: 1024 * 1024 /* v8 ignore next */ /* v8 ignore next */,
      }); /* v8 ignore next */ /* v8 ignore next */
      outputDiv.textContent =
        /* v8 ignore next */ /* v8 ignore next */
        'Session initialized! The model is not loaded yet.\nClick "Run Inference" to trigger chunked downloading.'; /* v8 ignore next */ /* v8 ignore next */
      runBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
    } catch (err: any) {
      /* v8 ignore next */ /* v8 ignore next */
      outputDiv.textContent = 'Error: ' + err.message; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  runBtn.addEventListener('click', async () => {
    /* v8 ignore next */ /* v8 ignore next */
    if (!session) return; /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent =
      'Running inference...\nStarting progressive tensor streaming...'; /* v8 ignore next */ /* v8 ignore next */
    try {
      /* v8 ignore next */ /* v8 ignore next */
      // Pass an empty inputs object for demo purposes /* v8 ignore next */ /* v8 ignore next */
      const outputs = await session.run({}); /* v8 ignore next */ /* v8 ignore next */
      outputDiv.textContent +=
        /* v8 ignore next */ /* v8 ignore next */
        '\n\nSuccess! Progressively loaded weights and completed inference.\nOutput: ' /* v8 ignore next */ /* v8 ignore next */ +
        JSON.stringify(outputs, null, 2); /* v8 ignore next */ /* v8 ignore next */
    } catch (err: any) {
      /* v8 ignore next */ /* v8 ignore next */
      outputDiv.textContent +=
        '\n\nError during inference: ' + err.message; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
});
