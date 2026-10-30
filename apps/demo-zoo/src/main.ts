/**
 * @fileoverview Model Zoo Progressive Safetensors Streaming Demo UI.
 */
import { fetchSafetensorsHeader, loadTensors } from '@onnx9000/core';

/**
 * Initializes the Model Zoo browser demo UI.
 */
export function initZooDemo(): void {
  const fetchBtn = document.getElementById('fetch-btn') as HTMLButtonElement | null;
  const streamBtn = document.getElementById('stream-btn') as HTMLButtonElement | null;
  const out = document.getElementById('zoo-output') as HTMLElement | null;
  const progressBar = document.getElementById('progress-bar') as HTMLElement | null;

  if (!fetchBtn || !streamBtn || !out || !progressBar) return;

  const MODEL_URL = 'hf://huggingface/co/bert-base-uncased/resolve/main/model.safetensors';

  fetchBtn.addEventListener('click', async () => {
    out.innerText = `Fetching Safetensors metadata from ${MODEL_URL}...
`;
    fetchBtn.disabled = true;

    try {
      const { headerObj, headerSize } = await fetchSafetensorsHeader(MODEL_URL);
      out.innerText += `
Successfully fetched metadata!`;
      out.innerText += `
Header Size: ${headerSize} bytes`;
      out.innerText += `
Tensors count: ${Object.keys(headerObj).length}`;

      if (headerObj.__metadata__ && 'format' in headerObj.__metadata__) {
        out.innerText += `
Format: ${headerObj.__metadata__.format || 'pt'}`;
      }

      out.innerText += `

Ready to stream weights progressively.`;
      streamBtn.disabled = false;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      out.innerText += `
Error: ${msg}`;
      fetchBtn.disabled = false;
    }
  });

  streamBtn.addEventListener('click', async () => {
    streamBtn.disabled = true;
    out.innerText += `

Starting progressive tensor streaming...`;
    progressBar.style.width = '0%';

    try {
      let count = 0;
      const limit = 2;

      for await (const tensor of loadTensors(MODEL_URL)) {
        count++;

        const p = Math.floor((count / limit) * 100);
        progressBar.style.width = `${p}%`;

        out.innerText += `
Loaded [${tensor.name}] dtype=${tensor.info.dtype} shape=[${tensor.info.shape.join(',')}]`;

        if (count >= limit) {
          break;
        }
      }
      out.innerText += `

Success! Progressively loaded weights using byte-range requests.`;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      out.innerText += `
Error: ${msg}`;
    } finally {
      streamBtn.disabled = false;
    }
  });
}

initZooDemo();
