/**
 * @fileoverview Main entry point for the ONNX to GGUF converter UI.
 * Handles model file upload, metadata extraction, user overrides, and GGUF serialization.
 */
import { load } from '@onnx9000/core';
import { extractMetadata, extractTokenizerMetadata, inferArchitecture } from '@onnx9000/onnx2gguf';

/** Interface `NavigatorWithMemory` definition. */
interface NavigatorWithMemory {
  /** Property `deviceMemory`. */
  deviceMemory?: number;
}

/** Interface `WritableFileStream` definition. */
interface WritableFileStream {
  /**
   * Method `write`.
   * @param data Parameter `data`.
   */
  write(data: BufferSource | Blob | string): Promise<void>;
  /** Method `close`. */
  close(): Promise<void>;
}

/** Interface `FileSystemFileHandle` definition. */
interface FileSystemFileHandle {
  /** Method `createWritable`. */
  createWritable(): Promise<WritableFileStream>;
}

/** Interface `WindowWithSaveFilePicker` definition. */
interface WindowWithSaveFilePicker {
  /** Property `showSaveFilePicker`. */
  showSaveFilePicker?: (options: {
    suggestedName: string;
    types: Array<{
      description: string;
      accept: Record<string, string[]>;
    }>;
  }) => Promise<FileSystemFileHandle>;
}

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

  const statusEl: HTMLElement = statusDiv;
  const metaTableEl: HTMLElement = metaTableBody;

  let modelBuffer: ArrayBuffer | null = null;
  let tokenizerStr: string | null = null;
  let graph: ReturnType<typeof JSON.parse> = null;
  let extractedMeta: Record<string, ReturnType<typeof JSON.parse>> = {};

  const navMem = (navigator as NavigatorWithMemory).deviceMemory;
  if (navMem && navMem < 8) {
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
    statusEl.textContent = 'Loading files...';
    for (const file of files) {
      if (file.name.endsWith('.onnx')) {
        modelBuffer = await file.arrayBuffer();
        statusEl.textContent = 'Parsing ONNX...';
        graph = await load(modelBuffer);
      } else if (file.name.endsWith('tokenizer.json')) {
        tokenizerStr = await file.text();
      }
    }

    if (graph) {
      statusEl.textContent = 'Extracting metadata...';
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
      statusEl.textContent = 'Ready for conversion.';
    } else {
      statusEl.textContent = 'Please provide an .onnx file.';
    }
  }

  function renderMetaTable() {
    metaTableEl.innerHTML = '';
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
      metaTableEl.appendChild(tr);
    }
  }

  convertBtn.addEventListener('click', async () => {
    if (!graph) {
      alert('Load an ONNX model first.');
      return;
    }

    extractedMeta['general.file_type'] = quantTarget.value;
    statusEl.textContent = 'Compiling GGUF (this may take a while)...';

    try {
      const { compileGGUF } = await import('@onnx9000/onnx2gguf');
      const t0 = performance.now();
      const buffer = compileGGUF(graph, extractedMeta);
      const t1 = performance.now();
      const speed = buffer.byteLength / 1024 / 1024 / ((t1 - t0) / 1000);

      statusEl.textContent = `Encoding speed: ${speed.toFixed(2)} MB/s`;

      const winPicker = window as WindowWithSaveFilePicker;
      if ('showSaveFilePicker' in window && winPicker.showSaveFilePicker) {
        const handle = await winPicker.showSaveFilePicker({
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
        statusEl.textContent += ' | Saved to disk via File System Access API.';
      } else {
        const blob = new Blob([buffer]);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'model.gguf';
        a.click();
        URL.revokeObjectURL(url);
        statusEl.textContent += ' | Downloaded.';
      }
    } catch (_e) {
      const e = _e instanceof Error ? _e : new Error(String(_e));
      statusEl.textContent = `Error: ${e.message}`;
      if (e.message.includes('memory') || e.message.includes('allocation')) {
        alert(`Hardware Constrained: ${e.message}`);
      }
    }
  });
}

initOnnx2GgufUI();
