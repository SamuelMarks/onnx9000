import { convert, SourceFramework, TargetFramework } from '@onnx9000/converters';
import { serializeModelProto } from '@onnx9000/core';

const srcFrameworkSelect = document.getElementById('src-framework') as HTMLSelectElement;
const dstFrameworkSelect = document.getElementById('dst-framework') as HTMLSelectElement;
const dropZone = document.getElementById('drop-zone') as HTMLDivElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const dropHint = document.getElementById('drop-hint') as HTMLParagraphElement;
const filesList = document.getElementById('files-list') as HTMLDivElement;
const btnConvert = document.getElementById('btn-convert') as HTMLButtonElement;
const btnDownload = document.getElementById('btn-download') as HTMLButtonElement;
const logsContainer = document.getElementById('logs') as HTMLDivElement;

// WebGL initialization check
try {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  if (!gl) {
    throw new Error('WebGL not supported');
  }
} catch (e) {
  alert('WebGL initialization failed. The previewer may not function correctly.');
  console.error('WebGL init error:', e);
}

const currentFiles: File[] = [];
let finalBlobUrl: string | null = null;
let finalFileName = 'model.onnx';

const frameworkRequirements: Record<string, { desc: string; check: (files: File[]) => boolean }> = {
  caffe: {
    desc: 'Requires: .prototxt and .caffemodel',
    check: (files) =>
      files.some((f) => f.name.endsWith('.prototxt')) &&
      files.some((f) => f.name.endsWith('.caffemodel')),
  },
  mxnet: {
    desc: 'Requires: -symbol.json and .params',
    check: (files) =>
      files.some((f) => f.name.endsWith('-symbol.json')) &&
      files.some((f) => f.name.endsWith('.params')),
  },
  cntk: {
    desc: 'Requires: .model',
    check: (files) => files.some((f) => f.name.endsWith('.model')),
  },
  darknet: {
    desc: 'Requires: .cfg and .weights',
    check: (files) =>
      files.some((f) => f.name.endsWith('.cfg')) && files.some((f) => f.name.endsWith('.weights')),
  },
  ncnn: {
    desc: 'Requires: .param and .bin',
    check: (files) =>
      files.some((f) => f.name.endsWith('.param')) && files.some((f) => f.name.endsWith('.bin')),
  },
  paddle: {
    desc: 'Requires: __model__ (or .pdmodel) and weights',
    check: (files) => files.some((f) => f.name === '__model__' || f.name.endsWith('.pdmodel')),
  },
  keras: {
    desc: 'Requires: .h5 or .keras',
    check: (files) => files.some((f) => f.name.endsWith('.h5') || f.name.endsWith('.keras')),
  },
  coreml: {
    desc: 'Requires: .mlmodel',
    check: (files) => files.some((f) => f.name.endsWith('.mlmodel')),
  },
};

function updateHint() {
  const req = frameworkRequirements[srcFrameworkSelect.value];
  if (req) {
    dropHint.textContent = req.desc;
  }
  validateFiles();
}

function updateFileList() {
  filesList.innerHTML = '';
  currentFiles.forEach((file, index) => {
    const item = document.createElement('div');
    item.className = 'file-item';
    item.innerHTML = `
      <span>${file.name} (${(file.size / 1024).toFixed(1)} KB)</span>
      <button type="button" style="padding: 0.2rem 0.5rem; flex: none;" data-index="${index}">X</button>
    `;
    const rmBtn = item.querySelector('button');
    rmBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      currentFiles.splice(index, 1);
      updateFileList();
    });
    filesList.appendChild(item);
  });
  validateFiles();
}

function validateFiles() {
  const req = frameworkRequirements[srcFrameworkSelect.value];
  if (req && req.check(currentFiles)) {
    btnConvert.disabled = false;
  } else {
    btnConvert.disabled = true;
  }
}

function log(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') {
  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  logsContainer.appendChild(line);
  logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Event Listeners
srcFrameworkSelect.addEventListener('change', updateHint);

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer?.files) {
    Array.from(e.dataTransfer.files).forEach((f) => currentFiles.push(f));
    updateFileList();
  }
});

dropZone.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener('change', (e) => {
  const target = e.target as HTMLInputElement;
  if (target.files) {
    Array.from(target.files).forEach((f) => currentFiles.push(f));
    updateFileList();
  }
  target.value = ''; // reset
});

btnConvert.addEventListener('click', async () => {
  btnConvert.disabled = true;
  btnDownload.classList.add('hidden');
  logsContainer.innerHTML = '';

  const src = srcFrameworkSelect.value as SourceFramework;
  const dst = dstFrameworkSelect.value as TargetFramework;

  log(`Starting conversion from ${src.toUpperCase()} to ${dst.toUpperCase()}...`, 'info');

  try {
    const result = await convert(src, dst, currentFiles, {
      fusion: true,
      shapeInference: true,
      layoutTracking: true,
      verbose: true,
    });

    let blob: Blob;
    let ext = '.onnx';
    if (dst === 'onnx') {
      const bytes = await serializeModelProto(result as any);
      blob = new Blob([bytes.buffer as ArrayBuffer], { type: 'application/octet-stream' });
    } else {
      if (dst === 'pytorch_code') ext = '.py';
      else if ((dst as string) === 'jax_code') ext = '.py';
      else if ((dst as string) === 'flax_nnx_code') ext = '.py';
      else if (dst === 'tfjs') ext = '_tfjs.json';
      else if (dst === 'coreml') ext = '.mlmodel';
      else if ((dst as string) === 'ncnn') ext = '.param';
      else ext = '.txt';

      blob = new Blob([typeof result === 'string' ? result : JSON.stringify(result, null, 2)], {
        type: 'text/plain',
      });
    }

    if (finalBlobUrl) {
      URL.revokeObjectURL(finalBlobUrl);
    }
    finalBlobUrl = URL.createObjectURL(blob);
    finalFileName = `converted_model${ext}`;

    log(`Conversion complete!`, 'success');
    btnDownload.textContent = `Download ${finalFileName}`;
    btnDownload.classList.remove('hidden');
    btnDownload.disabled = false;

    // Render basic summary to the graph preview
    const previewDiv = document.getElementById('graph-preview');
    if (previewDiv) {
      if (dst === 'onnx') {
        const nodeCount = (result as any).nodes?.length || 0;
        previewDiv.innerHTML = `<strong>ONNX Graph Generated</strong><br>Nodes: ${nodeCount}<br>Ready for download or 3D viewer.`;
        previewDiv.style.color = '#fff';
      } else {
        previewDiv.innerHTML = `<strong>Text Output Generated</strong><br>Lines: ${String(result).split('\\n').length}`;
        previewDiv.style.color = '#fff';
      }
    }
  } catch (err: any) {
    log(`Conversion failed: ${err.message}`, 'error');
  }

  btnConvert.disabled = false;
});

btnDownload.addEventListener('click', () => {
  if (finalBlobUrl) {
    const a = document.createElement('a');
    a.href = finalBlobUrl;
    a.download = finalFileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
});

// Init
updateHint();
