/**
 * @fileoverview JSON Extract Demo UI.
 */
import { load } from '@onnx9000/core';

/**
 * Initializes the JSON extract demo UI and sets up drag-and-drop / file handlers.
 */
export function initJsonExtractDemo(): void {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input') as HTMLInputElement | null;
  const browseBtn = document.getElementById('browse-btn') as HTMLButtonElement | null;
  const statusPanel = document.getElementById('status-panel');
  const resultPanel = document.getElementById('result-panel');
  const statusText = document.getElementById('status-text');
  const progressBar = document.getElementById('progress-bar');
  const errorBox = document.getElementById('error-box');
  const downloadBtn = document.getElementById('download-btn') as HTMLButtonElement | null;
  const statsText = document.getElementById('stats-text');

  if (
    !dropZone ||
    !fileInput ||
    !browseBtn ||
    !statusPanel ||
    !resultPanel ||
    !statusText ||
    !progressBar ||
    !errorBox ||
    !downloadBtn ||
    !statsText
  ) {
    return;
  }

  const validStatusPanel = statusPanel;
  const validResultPanel = resultPanel;
  const validStatusText = statusText;
  const validProgressBar = progressBar;
  const validErrorBox = errorBox;
  const validStatsText = statsText;

  let jsonBlob: Blob | null = null;
  let originalName = '';

  browseBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e: Event) => {
    const target = e.target as HTMLInputElement;
    const firstFile = target.files?.[0];
    if (firstFile) {
      void processFile(firstFile);
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
    const firstFile = e.dataTransfer?.files?.[0];
    if (firstFile) {
      void processFile(firstFile);
    }
  });

  async function processFile(file: File): Promise<void> {
    if (!file.name.endsWith('.onnx')) {
      showError('Please provide a valid .onnx file.');
      return;
    }
    originalName = file.name.replace('.onnx', '');
    validErrorBox.classList.add('hidden');
    validResultPanel.classList.add('hidden');
    validStatusPanel.classList.remove('hidden');

    try {
      await performExtraction(file);
    } catch (_err) {
      const err = _err instanceof Error ? _err : new Error(String(_err));
      showError(err.message);
    }
  }

  function updateProgress(msg: string, pct: number): void {
    validStatusText.textContent = msg;
    validProgressBar.style.width = `${pct}%`;
  }

  function showError(msg: string): void {
    validErrorBox.textContent = msg;
    validErrorBox.classList.remove('hidden');
    updateProgress('Failed', 100);
    validProgressBar.style.backgroundColor = '#cc3333';
  }

  async function performExtraction(file: File): Promise<void> {
    validProgressBar.style.backgroundColor = '#007acc';
    updateProgress('Reading file...', 10);

    const arrayBuffer = await file.arrayBuffer();
    updateProgress('Parsing ONNX AST...', 50);
    await new Promise((resolve) => setTimeout(resolve, 10));

    const t0 = performance.now();
    const graph = await load(arrayBuffer);
    updateProgress('Extracting JSON...', 80);
    await new Promise((resolve) => setTimeout(resolve, 10));

    const jsonString = JSON.stringify(
      graph,
      (key, value) => {
        if (key === 'data' && ArrayBuffer.isView(value)) {
          return `[Buffer: ${(value as ArrayBufferView).byteLength} bytes]`;
        }
        if (typeof value === 'bigint') {
          return `${value.toString()}n`;
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

    validStatsText.innerHTML = `<strong>File:</strong> ${originalName}.onnx (${inputSize} MB)<br/><strong>JSON Size:</strong> ${outputSize} MB<br/><strong>Nodes:</strong> ${graph.nodes.length}<br/><strong>Extraction Time:</strong> ${timeMs} ms`;
    validStatusPanel.classList.add('hidden');
    validResultPanel.classList.remove('hidden');
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
}

initJsonExtractDemo();
