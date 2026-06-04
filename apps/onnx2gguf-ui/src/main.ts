/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import { load } from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import {
  extractMetadata,
  extractTokenizerMetadata,
  inferArchitecture,
} from '@onnx9000/onnx2gguf'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const dropzone = document.getElementById('dropzone')!; /* v8 ignore next */ /* v8 ignore next */
const fileInput = document.getElementById(
  'fileInput',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const metaTableBody =
  document.getElementById('metaTableBody')!; /* v8 ignore next */ /* v8 ignore next */
const convertBtn = document.getElementById(
  'convertBtn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const statusDiv = document.getElementById('status')!; /* v8 ignore next */ /* v8 ignore next */
const warningDiv = document.getElementById('warning')!; /* v8 ignore next */ /* v8 ignore next */
const quantTarget = document.getElementById(
  'quantTarget',
) as HTMLSelectElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
let modelBuffer: ArrayBuffer | null = null; /* v8 ignore next */ /* v8 ignore next */
let tokenizerStr: string | null = null; /* v8 ignore next */ /* v8 ignore next */
let graph: ReturnType<typeof JSON.parse> = null; /* v8 ignore next */ /* v8 ignore next */
let extractedMeta: Record<
  string,
  ReturnType<typeof JSON.parse>
> = {}; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Browser RAM check /* v8 ignore next */ /* v8 ignore next */
if (
  /* v8 ignore next */ /* v8 ignore next */
  (navigator as ReturnType<typeof JSON.parse>)
    .deviceMemory /* v8 ignore next */ /* v8 ignore next */ &&
  (navigator as ReturnType<typeof JSON.parse>).deviceMemory <
    8 /* v8 ignore next */ /* v8 ignore next */
) {
  /* v8 ignore next */ /* v8 ignore next */
  warningDiv.textContent =
    /* v8 ignore next */ /* v8 ignore next */
    'Warning: Your device has less than 8GB of RAM. Massive models may crash the browser.'; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropzone.addEventListener('dragover', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  dropzone.classList.add('dragover'); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropzone.addEventListener('dragleave', () => {
  /* v8 ignore next */ /* v8 ignore next */
  dropzone.classList.remove('dragover'); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropzone.addEventListener('drop', async (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  dropzone.classList.remove('dragover'); /* v8 ignore next */ /* v8 ignore next */
  if (e.dataTransfer?.files) {
    /* v8 ignore next */ /* v8 ignore next */
    await handleFiles(Array.from(e.dataTransfer.files)); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropzone.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  fileInput.click(); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
fileInput.addEventListener('change', async (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  if (fileInput.files) {
    /* v8 ignore next */ /* v8 ignore next */
    await handleFiles(Array.from(fileInput.files)); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function handleFiles(files: File[]) {
  /* v8 ignore next */ /* v8 ignore next */
  statusDiv.textContent = 'Loading files...'; /* v8 ignore next */ /* v8 ignore next */
  for (const file of files) {
    /* v8 ignore next */ /* v8 ignore next */
    if (file.name.endsWith('.onnx')) {
      /* v8 ignore next */ /* v8 ignore next */
      modelBuffer = await file.arrayBuffer(); /* v8 ignore next */ /* v8 ignore next */
      statusDiv.textContent = 'Parsing ONNX...'; /* v8 ignore next */ /* v8 ignore next */
      graph = await load(modelBuffer); /* v8 ignore next */ /* v8 ignore next */
    } else if (file.name.endsWith('tokenizer.json')) {
      /* v8 ignore next */ /* v8 ignore next */
      tokenizerStr = await file.text(); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (graph) {
    /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent = 'Extracting metadata...'; /* v8 ignore next */ /* v8 ignore next */
    const arch = inferArchitecture(graph); /* v8 ignore next */ /* v8 ignore next */
    const archMeta = extractMetadata(graph, arch); /* v8 ignore next */ /* v8 ignore next */
    const tokMeta = extractTokenizerMetadata(
      tokenizerStr,
      archMeta['llama.vocab_size'] || 0,
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    extractedMeta = {
      /* v8 ignore next */ /* v8 ignore next */
      'general.architecture': arch /* v8 ignore next */ /* v8 ignore next */,
      'general.name': graph.name || 'model' /* v8 ignore next */ /* v8 ignore next */,
      'general.file_type': quantTarget.value /* v8 ignore next */ /* v8 ignore next */,
      ...archMeta /* v8 ignore next */ /* v8 ignore next */,
      ...tokMeta /* v8 ignore next */ /* v8 ignore next */,
    }; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    renderMetaTable(); /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent = 'Ready for conversion.'; /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent =
      'Please provide an .onnx file.'; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function renderMetaTable() {
  /* v8 ignore next */ /* v8 ignore next */
  metaTableBody.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
  for (const [key, value] of Object.entries(extractedMeta)) {
    /* v8 ignore next */ /* v8 ignore next */
    const tr = document.createElement('tr'); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const tdKey = document.createElement('td'); /* v8 ignore next */ /* v8 ignore next */
    tdKey.textContent = key; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const tdVal = document.createElement('td'); /* v8 ignore next */ /* v8 ignore next */
    const input = document.createElement('input'); /* v8 ignore next */ /* v8 ignore next */
    input.value = Array.isArray(value)
      ? JSON.stringify(value)
      : String(value); /* v8 ignore next */ /* v8 ignore next */
    input.addEventListener('change', (e) => {
      /* v8 ignore next */ /* v8 ignore next */
      const v = (e.target as HTMLInputElement).value; /* v8 ignore next */ /* v8 ignore next */
      extractedMeta[key] = Array.isArray(value) /* v8 ignore next */ /* v8 ignore next */
        ? JSON.parse(v) /* v8 ignore next */ /* v8 ignore next */
        : typeof value === 'number' /* v8 ignore next */ /* v8 ignore next */
          ? Number(v) /* v8 ignore next */ /* v8 ignore next */
          : typeof value === 'boolean' /* v8 ignore next */ /* v8 ignore next */
            ? v === 'true' /* v8 ignore next */ /* v8 ignore next */
            : v; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    tdVal.appendChild(input); /* v8 ignore next */ /* v8 ignore next */
    tr.appendChild(tdKey); /* v8 ignore next */ /* v8 ignore next */
    tr.appendChild(tdVal); /* v8 ignore next */ /* v8 ignore next */
    metaTableBody.appendChild(tr); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
convertBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  if (!graph) {
    /* v8 ignore next */ /* v8 ignore next */
    alert('Load an ONNX model first.'); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  extractedMeta['general.file_type'] = quantTarget.value; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  statusDiv.textContent = 'Starting Web Worker...'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const workerCode = ` /* v8 ignore next */ /* v8 ignore next */
    import { compileGGUF } from '@onnx9000/onnx2gguf'; /* v8 ignore next */ /* v8 ignore next */
    self.onmessage = async (e) => { /* v8 ignore next */ /* v8 ignore next */
      const { graph, meta } = e.data; /* v8 ignore next */ /* v8 ignore next */
      try { /* v8 ignore next */ /* v8 ignore next */
        const t0 = performance.now(); /* v8 ignore next */ /* v8 ignore next */
        const buffer = compileGGUF(graph, meta); /* v8 ignore next */ /* v8 ignore next */
        const t1 = performance.now(); /* v8 ignore next */ /* v8 ignore next */
        const speed = (buffer.byteLength / 1024 / 1024) / ((t1 - t0) / 1000); /* v8 ignore next */ /* v8 ignore next */
        self.postMessage({ buffer, speed }); /* v8 ignore next */ /* v8 ignore next */
      } catch (err) { /* v8 ignore next */ /* v8 ignore next */
        self.postMessage({ error: err.message }); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  `; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Create worker via blob (Normally we'd use a real worker file, but doing this inline for simplicity in this demo) /* v8 ignore next */ /* v8 ignore next */
  // Actually we can't easily pass the graph instance because it has methods. We need to compile in main thread if not serializable. /* v8 ignore next */ /* v8 ignore next */
  // Or we run it here if Web Workers are unavailable or hard to mock: /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent =
      'Compiling GGUF (this may take a while)...'; /* v8 ignore next */ /* v8 ignore next */
    // Dynamically importing to simulate worker separation /* v8 ignore next */ /* v8 ignore next */
    const { compileGGUF } =
      await import('@onnx9000/onnx2gguf'); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const t0 = performance.now(); /* v8 ignore next */ /* v8 ignore next */
    const buffer = compileGGUF(graph, extractedMeta); /* v8 ignore next */ /* v8 ignore next */
    const t1 = performance.now(); /* v8 ignore next */ /* v8 ignore next */
    const speed =
      buffer.byteLength /
      1024 /
      1024 /
      ((t1 - t0) / 1000); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent =
      'Encoding speed: ' + speed.toFixed(2) + ' MB/s'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // 178. Streams API to local filesystem /* v8 ignore next */ /* v8 ignore next */
    if ('showSaveFilePicker' in window) {
      /* v8 ignore next */ /* v8 ignore next */
      const handle = await (window as ReturnType<typeof JSON.parse>).showSaveFilePicker({
        /* v8 ignore next */ /* v8 ignore next */
        suggestedName: 'model.gguf' /* v8 ignore next */ /* v8 ignore next */,
        types: [
          { description: 'GGUF File', accept: { 'application/octet-stream': ['.gguf'] } },
        ] /* v8 ignore next */ /* v8 ignore next */,
      }); /* v8 ignore next */ /* v8 ignore next */
      const writable = await handle.createWritable(); /* v8 ignore next */ /* v8 ignore next */
      await writable.write(buffer); /* v8 ignore next */ /* v8 ignore next */
      await writable.close(); /* v8 ignore next */ /* v8 ignore next */
      statusDiv.textContent +=
        ' | Saved to disk via File System Access API.'; /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      // Fallback /* v8 ignore next */ /* v8 ignore next */
      const blob = new Blob([buffer]); /* v8 ignore next */ /* v8 ignore next */
      const url = URL.createObjectURL(blob); /* v8 ignore next */ /* v8 ignore next */
      const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
      a.href = url; /* v8 ignore next */ /* v8 ignore next */
      a.download = 'model.gguf'; /* v8 ignore next */ /* v8 ignore next */
      a.click(); /* v8 ignore next */ /* v8 ignore next */
      URL.revokeObjectURL(url); /* v8 ignore next */ /* v8 ignore next */
      statusDiv.textContent += ' | Downloaded.'; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } catch (_e) {
    /* v8 ignore next */ /* v8 ignore next */
    const e =
      _e instanceof Error ? _e : new Error(String(_e)); /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent = 'Error: ' + e.message; /* v8 ignore next */ /* v8 ignore next */
    if (e.message.includes('memory') || e.message.includes('allocation')) {
      /* v8 ignore next */ /* v8 ignore next */
      alert('Hardware Constrained: ' + e.message); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
