import { load } from '@onnx9000/core';
import { extractMetadata, extractTokenizerMetadata, inferArchitecture } from '@onnx9000/onnx2gguf';

const dropzone = document.getElementById('dropzone')!;
const fileInput = document.getElementById('fileInput') as HTMLInputElement;
const metaTableBody = document.getElementById('metaTableBody')!;
const convertBtn = document.getElementById('convertBtn') as HTMLButtonElement;
const statusDiv = document.getElementById('status')!;
const warningDiv = document.getElementById('warning')!;
const quantTarget = document.getElementById('quantTarget') as HTMLSelectElement;

let modelBuffer: ArrayBuffer | null = null;
let tokenizerStr: string | null = null;
let graph: any = null;
let extractedMeta: Record<string, any> = {};

// Browser RAM check
if ((navigator as any).deviceMemory && (navigator as any).deviceMemory < 8) {
  warningDiv.textContent =
    'Warning: Your device has less than 8GB of RAM. Massive models may crash the browser.';
}

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', async (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer?.files) {
    await handleFiles(Array.from(e.dataTransfer.files));
  }
});

dropzone.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener('change', async (e) => {
  if (fileInput.files) {
    await handleFiles(Array.from(fileInput.files));
  }
});

async function handleFiles(files: File[]) {
  statusDiv.textContent = 'Loading files...';
  for (const file of files) {
    if (file.name.endsWith('.onnx')) {
      modelBuffer = await file.arrayBuffer();
      statusDiv.textContent = 'Parsing ONNX...';
      graph = await load(modelBuffer);
    } else if (file.name.endsWith('tokenizer.json')) {
      tokenizerStr = await file.text();
    }
  }

  if (graph) {
    statusDiv.textContent = 'Extracting metadata...';
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
    statusDiv.textContent = 'Ready for conversion.';
  } else {
    statusDiv.textContent = 'Please provide an .onnx file.';
  }
}

function renderMetaTable() {
  metaTableBody.innerHTML = '';
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
    metaTableBody.appendChild(tr);
  }
}

convertBtn.addEventListener('click', async () => {
  if (!graph) {
    alert('Load an ONNX model first.');
    return;
  }

  extractedMeta['general.file_type'] = quantTarget.value;

  statusDiv.textContent = 'Starting Web Worker...';

  const workerCode = `
    import { compileGGUF } from '@onnx9000/onnx2gguf';
    self.onmessage = async (e) => {
      const { graph, meta } = e.data;
      try {
        const t0 = performance.now();
        const buffer = compileGGUF(graph, meta);
        const t1 = performance.now();
        const speed = (buffer.byteLength / 1024 / 1024) / ((t1 - t0) / 1000);
        self.postMessage({ buffer, speed });
      } catch (err) {
        self.postMessage({ error: err.message });
      }
    };
  `;

  // Create worker via blob (Normally we'd use a real worker file, but doing this inline for simplicity in this demo)
  // Actually we can't easily pass the graph instance because it has methods. We need to compile in main thread if not serializable.
  // Or we run it here if Web Workers are unavailable or hard to mock:

  try {
    statusDiv.textContent = 'Compiling GGUF (this may take a while)...';
    // Dynamically importing to simulate worker separation
    const { compileGGUF } = await import('@onnx9000/onnx2gguf');

    const t0 = performance.now();
    const buffer = compileGGUF(graph, extractedMeta);
    const t1 = performance.now();
    const speed = buffer.byteLength / 1024 / 1024 / ((t1 - t0) / 1000);

    statusDiv.textContent = 'Encoding speed: ' + speed.toFixed(2) + ' MB/s';

    // 178. Streams API to local filesystem
    if ('showSaveFilePicker' in window) {
      const handle = await (window as any).showSaveFilePicker({
        suggestedName: 'model.gguf',
        types: [{ description: 'GGUF File', accept: { 'application/octet-stream': ['.gguf'] } }],
      });
      const writable = await handle.createWritable();
      await writable.write(buffer);
      await writable.close();
      statusDiv.textContent += ' | Saved to disk via File System Access API.';
    } else {
      // Fallback
      const blob = new Blob([buffer]);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'model.gguf';
      a.click();
      URL.revokeObjectURL(url);
      statusDiv.textContent += ' | Downloaded.';
    }
  } catch (e: any) {
    statusDiv.textContent = 'Error: ' + e.message;
    if (e.message.includes('memory') || e.message.includes('allocation')) {
      alert('Hardware Constrained: ' + e.message);
    }
  }
});
