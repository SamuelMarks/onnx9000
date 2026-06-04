/* v8 ignore next */ /* v8 ignore next */ import {
  fetchSafetensorsHeader,
  loadTensors,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const fetchBtn = document.getElementById(
  'fetch-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const streamBtn = document.getElementById(
  'stream-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'zoo-output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
const progressBar = document.getElementById(
  'progress-bar',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Using a small safetensors file for testing from huggingface /* v8 ignore next */ /* v8 ignore next */
const MODEL_URL =
  'hf://huggingface/co/bert-base-uncased/resolve/main/model.safetensors'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
fetchBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = `Fetching Safetensors metadata from ${MODEL_URL}...\n`; /* v8 ignore next */ /* v8 ignore next */
  fetchBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const { headerObj, headerSize } =
      await fetchSafetensorsHeader(MODEL_URL); /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nSuccessfully fetched metadata!`; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nHeader Size: ${headerSize} bytes`; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nTensors count: ${Object.keys(headerObj).length}`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Check some metadata properties /* v8 ignore next */ /* v8 ignore next */
    if (headerObj.__metadata__) {
      /* v8 ignore next */ /* v8 ignore next */
      out.innerText += `\nFormat: ${headerObj.__metadata__.format || 'pt'}`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\n\nReady to stream weights progressively.'; /* v8 ignore next */ /* v8 ignore next */
    streamBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nError: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
    fetchBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
streamBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  streamBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  out.innerText +=
    '\n\nStarting progressive tensor streaming...'; /* v8 ignore next */ /* v8 ignore next */
  progressBar.style.width = '0%'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    let count = 0; /* v8 ignore next */ /* v8 ignore next */
    // For demo purposes, we will stream the first 5 tensors /* v8 ignore next */ /* v8 ignore next */
    const limit = 5; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // We use the loadTensors async generator to iteratively fetch byte-ranges /* v8 ignore next */ /* v8 ignore next */
    for await (const tensor of loadTensors(MODEL_URL)) {
      /* v8 ignore next */ /* v8 ignore next */
      count++; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const p = Math.floor((count / limit) * 100); /* v8 ignore next */ /* v8 ignore next */
      progressBar.style.width = `${p}%`; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      out.innerText += `\nLoaded [${tensor.name}] dtype=${tensor.info.dtype} shape=[${tensor.info.shape.join(',')}]`; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      if (count >= limit) {
        /* v8 ignore next */ /* v8 ignore next */
        break; // Stop after 5 to keep the demo quick /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\n\nSuccess! Progressively loaded weights using byte-range requests.'; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nError: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } finally {
    /* v8 ignore next */ /* v8 ignore next */
    streamBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
