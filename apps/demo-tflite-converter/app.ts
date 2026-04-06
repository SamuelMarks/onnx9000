import { parseModelProto, BufferReader } from '@onnx9000/core';
import { TFLiteExporter, compileGraphToTFLite } from '@onnx9000/tflite-exporter';

const dropZone = document.getElementById('drop-zone') as HTMLDivElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const browseBtn = document.getElementById('browse-btn') as HTMLButtonElement;
const statusPanel = document.getElementById('status-panel') as HTMLDivElement;
const resultPanel = document.getElementById('result-panel') as HTMLDivElement;
const statusText = document.getElementById('status-text') as HTMLParagraphElement;
const progressBar = document.getElementById('progress-bar') as HTMLDivElement;
const errorBox = document.getElementById('error-box') as HTMLDivElement;
const downloadBtn = document.getElementById('download-btn') as HTMLButtonElement;
const statsText = document.getElementById('stats-text') as HTMLParagraphElement;

const optEdgeTpu = document.getElementById('opt-edgetpu') as HTMLInputElement;
const quantFp16 = document.getElementById('quant-fp16') as HTMLInputElement;
const quantInt8 = document.getElementById('quant-int8') as HTMLInputElement;

let currentFile: File | null = null;
let tfliteBlob: Blob | null = null;
let originalName = '';

browseBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
  const target = e.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    handleFile(target.files[0]!);
  }
});

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
  if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]!);
  }
});

async function handleFile(file: File) {
  if (!file.name.endsWith('.onnx')) {
    showError('Please provide a valid .onnx file.');
    return;
  }

  currentFile = file;
  originalName = file.name.replace('.onnx', '');
  errorBox.classList.add('hidden');
  resultPanel.classList.add('hidden');
  statusPanel.classList.remove('hidden');

  try {
    await processModel();
  } catch (_err) {
    const err = _err instanceof Error ? _err : new Error(String(_err));
    showError(err.message || err.toString());
  }
}

function updateStatus(msg: string, progress: number) {
  statusText.textContent = msg;
  progressBar.style.width = `${progress}%`;
}

function showError(msg: string) {
  errorBox.textContent = msg;
  errorBox.classList.remove('hidden');
  updateStatus('Failed', 100);
  progressBar.style.backgroundColor = 'var(--error)';
}

async function processModel() {
  if (!currentFile) return;

  progressBar.style.backgroundColor = 'var(--accent)';
  updateStatus('Reading file...', 10);

  const buffer = await currentFile.arrayBuffer();

  updateStatus('Parsing ONNX AST...', 30);
  // Give UI a tick to render
  await new Promise((r) => setTimeout(r, 10));

  const reader = new BufferReader(new Uint8Array(buffer));
  const graph = await parseModelProto(reader);

  updateStatus('Optimizing layout & generating TFLite FlatBuffer...', 60);
  await new Promise((r) => setTimeout(r, 10));

  let quantMode: 'none' | 'fp16' | 'int8' = 'none';
  if (quantFp16.checked) quantMode = 'fp16';
  if (quantInt8.checked) quantMode = 'int8';

  const keepNchw = !optEdgeTpu.checked;
  const exporter = new TFLiteExporter();

  const t0 = performance.now();
  const subgraphsOffset = compileGraphToTFLite(graph, exporter, keepNchw, quantMode);

  updateStatus('Serializing FlatBuffer structures...', 80);
  await new Promise((r) => setTimeout(r, 10));

  exporter.builder.startVector(4, 1, 4);
  exporter.builder.addOffset(subgraphsOffset);
  const subgraphsVecOffset = exporter.builder.endVector(1);

  const tfliteBytes = exporter.finish(subgraphsVecOffset, `onnx9000-web-${originalName}`);
  const t1 = performance.now();

  updateStatus('Done!', 100);

  tfliteBlob = new Blob([tfliteBytes], { type: 'application/octet-stream' });

  const mbOriginal = (buffer.byteLength / 1024 / 1024).toFixed(2);
  const mbNew = (tfliteBytes.byteLength / 1024 / 1024).toFixed(2);
  const timeMs = (t1 - t0).toFixed(0);

  statsText.innerHTML = `
    <strong>Original Size:</strong> ${mbOriginal} MB<br>
    <strong>TFLite Size:</strong> ${mbNew} MB<br>
    <strong>Compilation Time:</strong> ${timeMs} ms
  `;

  resultPanel.classList.remove('hidden');
}

const openNetronCheckbox = document.getElementById('open-netron') as HTMLInputElement;

downloadBtn.addEventListener('click', () => {
  if (!tfliteBlob) return;
  const url = URL.createObjectURL(tfliteBlob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${originalName}.tflite`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  if (openNetronCheckbox.checked) {
    // 296. Offer an embedded interactive graph visualizer (Netron style) showing the final TFLite layout.
    // Open the local dev-server endpoint for Netron UI (if we are in the monorepo context) or public Netron.
    const netronUrl = `https://netron.app/?url=${encodeURIComponent(url)}`;
    window.open(netronUrl, '_blank');
  }

  // Revoke after a delay to allow the new window to fetch if needed
  setTimeout(() => URL.revokeObjectURL(url), 10000);
});
