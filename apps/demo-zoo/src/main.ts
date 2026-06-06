/* v8 ignore start */
import { fetchSafetensorsHeader, loadTensors } from "@onnx9000/core";

/**
 * Initializes the Model Zoo browser demo UI.
 */
export function initZooDemo(): void {
  const fetchBtn = document.getElementById("fetch-btn") as HTMLButtonElement;
  const streamBtn = document.getElementById("stream-btn") as HTMLButtonElement;
  const out = document.getElementById("zoo-output") as HTMLElement;
  const progressBar = document.getElementById("progress-bar") as HTMLElement;

  if (!fetchBtn || !streamBtn || !out || !progressBar) return;

  const MODEL_URL =
    "hf://huggingface/co/bert-base-uncased/resolve/main/model.safetensors";

  fetchBtn.addEventListener("click", async () => {
    out.innerText = `Fetching Safetensors metadata from ${MODEL_URL}...\n`;
    fetchBtn.disabled = true;

    try {
      const { headerObj, headerSize } = await fetchSafetensorsHeader(MODEL_URL);
      out.innerText += `\nSuccessfully fetched metadata!`;
      out.innerText += `\nHeader Size: ${headerSize} bytes`;
      out.innerText += `\nTensors count: ${Object.keys(headerObj).length}`;

      if (headerObj.__metadata__) {
        out.innerText += `\nFormat: ${headerObj.__metadata__.format || "pt"}`;
      }

      out.innerText += "\n\nReady to stream weights progressively.";
      streamBtn.disabled = false;
    } catch (e: any) {
      out.innerText += `\nError: ${e.message}`;
      fetchBtn.disabled = false;
    }
  });

  streamBtn.addEventListener("click", async () => {
    streamBtn.disabled = true;
    out.innerText += "\n\nStarting progressive tensor streaming...";
    progressBar.style.width = "0%";

    try {
      let count = 0;
      const limit = 5;

      for await (const tensor of loadTensors(MODEL_URL)) {
        count++;

        const p = Math.floor((count / limit) * 100);
        progressBar.style.width = `${p}%`;

        out.innerText += `\nLoaded [${tensor.name}] dtype=${tensor.info.dtype} shape=[${tensor.info.shape.join(",")}]`;

        if (count >= limit) {
          break;
        }
      }
      out.innerText +=
        "\n\nSuccess! Progressively loaded weights using byte-range requests.";
    } catch (e: any) {
      out.innerText += `\nError: ${e.message}`;
    } finally {
      streamBtn.disabled = false;
    }
  });
}

initZooDemo();

/* v8 ignore stop */
