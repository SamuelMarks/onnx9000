/**
 * @fileoverview TFLite FlatBuffer Converter Demo UI.
 */
import { BufferReader, parseModelProto } from '@onnx9000/core';
import { compileGraphToTFLite, TFLiteExporter } from '@onnx9000/tflite-exporter';

/**
 * Initializes the TFLite converter demo UI.
 */
export function initTfliteConverterDemo(): void {
  const dropZone = document.getElementById('drop-zone') as HTMLDivElement | null;
  const fileInput = document.getElementById('file-input') as HTMLInputElement | null;
  const browseBtn = document.getElementById('browse-btn') as HTMLButtonElement | null;
  const statusPanel = document.getElementById('status-panel') as HTMLDivElement | null;
  const resultPanel = document.getElementById('result-panel') as HTMLDivElement | null;
  const statusText = document.getElementById('status-text') as HTMLParagraphElement | null;
  const progressBar = document.getElementById('progress-bar') as HTMLDivElement | null;
  const errorBox = document.getElementById('error-box') as HTMLDivElement | null;
  const downloadBtn = document.getElementById('download-btn') as HTMLButtonElement | null;
  const statsText = document.getElementById('stats-text') as HTMLParagraphElement | null;
  const optEdgeTpu = document.getElementById('opt-edgetpu') as HTMLInputElement | null;
  const quantFp16 = document.getElementById('quant-fp16') as HTMLInputElement | null;
  const quantInt8 = document.getElementById('quant-int8') as HTMLInputElement | null;
  const openNetronCheckbox = document.getElementById('open-netron') as HTMLInputElement | null;

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
    !statsText ||
    !optEdgeTpu ||
    !quantFp16 ||
    !quantInt8 ||
    !openNetronCheckbox
  ) {
    return;
  }

  const validStatusPanel = statusPanel;
  const validResultPanel = resultPanel;
  const validStatusText = statusText;
  const validProgressBar = progressBar;
  const validErrorBox = errorBox;
  const validStatsText = statsText;
  const validOptEdgeTpu = optEdgeTpu;
  const validQuantFp16 = quantFp16;
  const validQuantInt8 = quantInt8;
  const validOpenNetron = openNetronCheckbox;

  let tfliteBlob: Blob | null = null;
  let originalName = '';

  browseBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    const target = e.target as HTMLInputElement;
    const firstFile = target.files?.[0];
    if (firstFile) {
      void handleFile(firstFile);
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
    const firstFile = e.dataTransfer?.files?.[0];
    if (firstFile) {
      void handleFile(firstFile);
    }
  });

  async function handleFile(file: File): Promise<void> {
    if (!file.name.endsWith('.onnx')) {
      showError('Please provide a valid .onnx file.');
      return;
    }

    originalName = file.name.replace('.onnx', '');
    validErrorBox.classList.add('hidden');
    validResultPanel.classList.add('hidden');
    validStatusPanel.classList.remove('hidden');

    try {
      await processModel(file);
    } catch (_err) {
      const err = _err instanceof Error ? _err : new Error(String(_err));
      showError(err.message);
    }
  }

  function updateStatus(msg: string, progress: number): void {
    validStatusText.textContent = msg;
    validProgressBar.style.width = `${progress}%`;
  }

  function showError(msg: string): void {
    validErrorBox.textContent = msg;
    validErrorBox.classList.remove('hidden');
    updateStatus('Failed', 100);
    validProgressBar.style.backgroundColor = 'var(--error)';
  }

  async function processModel(file: File): Promise<void> {
    validProgressBar.style.backgroundColor = 'var(--accent)';
    updateStatus('Reading file...', 10);

    const buffer = await file.arrayBuffer();

    updateStatus('Parsing ONNX AST...', 30);
    await new Promise((r) => setTimeout(r, 10));

    const reader = new BufferReader(new Uint8Array(buffer));
    const graph = await parseModelProto(reader);

    updateStatus('Optimizing layout & generating TFLite FlatBuffer...', 60);
    await new Promise((r) => setTimeout(r, 10));

    let quantMode: 'none' | 'fp16' | 'int8' = 'none';
    if (validQuantFp16.checked) quantMode = 'fp16';
    if (validQuantInt8.checked) quantMode = 'int8';

    const keepNchw = !validOptEdgeTpu.checked;
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

    tfliteBlob = new Blob([tfliteBytes.slice().buffer], {
      type: 'application/octet-stream',
    });

    const mbOriginal = (buffer.byteLength / 1024 / 1024).toFixed(2);
    const mbNew = (tfliteBytes.byteLength / 1024 / 1024).toFixed(2);
    const timeMs = (t1 - t0).toFixed(0);

    validStatsText.innerHTML = `  
    <strong>Original Size:</strong> ${mbOriginal} MB<br>  
    <strong>TFLite Size:</strong> ${mbNew} MB<br>  
    <strong>Compilation Time:</strong> ${timeMs} ms  
  `;

    validResultPanel.classList.remove('hidden');
  }

  downloadBtn.addEventListener('click', () => {
    if (!tfliteBlob) return;
    const url = URL.createObjectURL(tfliteBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${originalName}.tflite`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    if (validOpenNetron.checked) {
      const netronUrl = `https://netron.app/?url=${encodeURIComponent(url)}`;
      window.open(netronUrl, '_blank');
    }

    setTimeout(() => URL.revokeObjectURL(url), 50);
  });
}

initTfliteConverterDemo();
