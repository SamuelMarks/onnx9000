import { load } from '@onnx9000/core';

const dropZone = document.getElementById('drop-zone') as HTMLElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const browseBtn = document.getElementById('browse-btn') as HTMLButtonElement;

const statusPanel = document.getElementById('status-panel') as HTMLElement;
const resultPanel = document.getElementById('result-panel') as HTMLElement;
const statusText = document.getElementById('status-text') as HTMLElement;
const progressBar = document.getElementById('progress-bar') as HTMLElement;
const errorBox = document.getElementById('error-box') as HTMLElement;
const downloadBtn = document.getElementById('download-btn') as HTMLButtonElement;
const statsText = document.getElementById('stats-text') as HTMLElement;

let currentFile: File | null = null;
let jsonBlob: Blob | null = null;
let originalName: string = '';

browseBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e: Event) => {
  const target = e.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    processFile(target.files[0]);
  }
});

dropZone.addEventListener('dragover', (e: DragEvent) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e: DragEvent) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
    processFile(e.dataTransfer.files[0]);
  }
});

async function processFile(file: File) {
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
    await performExtraction();
  } catch (_err) {
    const err = _err instanceof Error ? _err : new Error(String(_err));
    showError(err.message || err.toString());
  }
}

function updateProgress(msg: string, pct: number) {
  statusText.textContent = msg;
  progressBar.style.width = `${pct}%`;
}

function showError(msg: string) {
  errorBox.textContent = msg;
  errorBox.classList.remove('hidden');
  updateProgress('Failed', 100);
  progressBar.style.backgroundColor = '#cc3333';
}

async function performExtraction() {
  if (!currentFile) return;

  progressBar.style.backgroundColor = '#007acc';
  updateProgress('Reading file...', 10);

  const arrayBuffer = await currentFile.arrayBuffer();

  updateProgress('Parsing ONNX AST...', 50);
  await new Promise((resolve) => setTimeout(resolve, 10)); // Yield to paint

  const t0 = performance.now();
  const graph = await load(arrayBuffer);

  updateProgress('Extracting JSON...', 80);
  await new Promise((resolve) => setTimeout(resolve, 10));

  const jsonString = JSON.stringify(
    graph,
    (key, value) => {
      // Drop heavy raw data for UI performance, just keep shapes and metadata
      if (key === 'data' && ArrayBuffer.isView(value)) {
        return `[Buffer: ${value.byteLength} bytes]`;
      }
      // Handle bigints
      if (typeof value === 'bigint') {
        return value.toString() + 'n';
      }
      return value;
    },
    2,
  );
  const t1 = performance.now();

  updateProgress('Done!', 100);

  jsonBlob = new Blob([jsonString], { type: 'application/json' });

  const inputSize = (arrayBuffer.byteLength / 1024 / 1024).toFixed(2);
  const outputSize = (jsonBlob.size / 1024 / 1024).toFixed(2);
  const timeMs = (t1 - t0).toFixed(0);

  statsText.innerHTML = `
    <strong>File:</strong> ${originalName}.onnx (${inputSize} MB)<br/>
    <strong>JSON Size:</strong> ${outputSize} MB<br/>
    <strong>Nodes:</strong> ${graph.nodes.length}<br/>
    <strong>Extraction Time:</strong> ${timeMs} ms
  `;

  statusPanel.classList.add('hidden');
  resultPanel.classList.remove('hidden');
}

downloadBtn.addEventListener('click', () => {
  if (!jsonBlob) return;
  const url = URL.createObjectURL(jsonBlob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `onnx9000-extracted-${originalName}.json`;
  a.click();
  URL.revokeObjectURL(url);
});
