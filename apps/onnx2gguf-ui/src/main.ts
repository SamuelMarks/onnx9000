/**
 * @fileoverview Main entry point for the ONNX to GGUF converter UI.
 * Handles model file upload, metadata extraction, user overrides, and GGUF serialization.
 */
import { load } from '@onnx9000/core';
import { extractMetadata, extractTokenizerMetadata, inferArchitecture } from '@onnx9000/onnx2gguf';

/**
 * Initializes the ONNX2GGUF UI.
 */
export function initOnnx2GgufUI(): void {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput') as HTMLInputElement;
  const metaTableBody = document.getElementById('metaTableBody');
  const convertBtn = document.getElementById('convertBtn') as HTMLButtonElement;
  const statusDiv = document.getElementById('status');
  const warningDiv = document.getElementById('warning');
  const quantTarget = document.getElementById('quantTarget') as HTMLSelectElement;

  if (
    !dropzone ||
    !fileInput ||
    !metaTableBody ||
    !convertBtn ||
    !statusDiv ||
    !warningDiv ||
    !quantTarget
  )
    return;

  let modelBuffer: ArrayBuffer | null = null;
  let tokenizerStr: string | null = null;
  let graph: ReturnType<typeof JSON.parse> = null;
  let extractedMeta: Record<string, ReturnType<typeof JSON.parse>> = {};

  if ((navigator as any).deviceMemory && (navigator as any).deviceMemory < 8) {
    warningDiv.textContent =
      'Warning: Your device has less than 8GB of RAM. Massive models may crash the browser.';
  }

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer?.files) {
      await handleFiles(Array.from(e.dataTransfer.files));
    }
  });

  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', async () => {
    if (fileInput.files) await handleFiles(Array.from(fileInput.files));
  });

  async function handleFiles(files: File[]) {
    statusDiv!.textContent = 'Loading files...';
    for (const file of files) {
      if (file.name.endsWith('.onnx')) {
        modelBuffer = await file.arrayBuffer();
        statusDiv!.textContent = 'Parsing ONNX...';
        graph = await load(modelBuffer);
      } else if (file.name.endsWith('tokenizer.json')) {
        tokenizerStr = await file.text();
      }
    }

    if (graph) {
      statusDiv!.textContent = 'Extracting metadata...';
      const arch = inferArchitecture(graph);
      const archMeta = extractMetadata(graph, arch);
      const tokMeta = extractTokenizerMetadata(tokenizerStr, archMeta['llama.vocab_size'] || 0);

      extractedMeta = {
        'general.architecture': arch,
        'general.name': graph.name || 'model',
        'general.file_type': quantTarget.value,
        ...archMeta,
        ...tokMeta,
      };

      renderMetaTable();
      statusDiv!.textContent = 'Ready for conversion.';
    } else {
      statusDiv!.textContent = 'Please provide an .onnx file.';
    }
  }

  function renderMetaTable() {
    metaTableBody!.innerHTML = '';
    for (const [key, value] of Object.entries(extractedMeta)) {
      const tr = document.createElement('tr');
      const tdKey = document.createElement('td');
      tdKey.textContent = key;
      const tdVal = document.createElement('td');
      const input = document.createElement('input');
      input.value = Array.isArray(value) ? JSON.stringify(value) : String(value);
      input.addEventListener('change', (e) => {
        const v = (e.target as HTMLInputElement).value;
        extractedMeta[key] = Array.isArray(value)
          ? JSON.parse(v)
          : typeof value === 'number'
            ? Number(v)
            : typeof value === 'boolean'
              ? v === 'true'
              : v;
      });
      tdVal.appendChild(input);
      tr.appendChild(tdKey);
      tr.appendChild(tdVal);
      metaTableBody?.appendChild(tr);
    }
  }

  convertBtn.addEventListener('click', async () => {
    if (!graph) {
      alert('Load an ONNX model first.');
      return;
    }

    extractedMeta['general.file_type'] = quantTarget.value;
    statusDiv!.textContent = 'Compiling GGUF (this may take a while)...';

    try {
      const { compileGGUF } = await import('@onnx9000/onnx2gguf');
      const t0 = performance.now();
      const buffer = compileGGUF(graph, extractedMeta);
      const t1 = performance.now();
      const speed = buffer.byteLength / 1024 / 1024 / ((t1 - t0) / 1000);

      statusDiv!.textContent = `Encoding speed: ${speed.toFixed(2)} MB/s`;

      if ('showSaveFilePicker' in window) {
        const handle = await (window as any).showSaveFilePicker({
          suggestedName: 'model.gguf',
          types: [
            {
              description: 'GGUF File',
              accept: { 'application/octet-stream': ['.gguf'] },
            },
          ],
        });
        const writable = await handle.createWritable();
        await writable.write(buffer);
        await writable.close();
        statusDiv!.textContent += ' | Saved to disk via File System Access API.';
      } else {
        const blob = new Blob([buffer]);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'model.gguf';
        a.click();
        URL.revokeObjectURL(url);
        statusDiv!.textContent += ' | Downloaded.';
      }
    } catch (_e) {
      const e = _e instanceof Error ? _e : new Error(String(_e));
      statusDiv!.textContent = `Error: ${e.message}`;
      if (e.message.includes('memory') || e.message.includes('allocation')) {
        alert(`Hardware Constrained: ${e.message}`);
      }
    }
  });
}

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', initOnnx2GgufUI);
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    initOnnx2GgufUI();
  }
}
