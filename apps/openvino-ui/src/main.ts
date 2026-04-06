import { load } from '@onnx9000/core';
import { OpenVinoExporter } from '@onnx9000/openvino-exporter';
import JSZip from 'jszip';

const dropzone = document.getElementById('dropzone')!;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const statusDiv = document.getElementById('status')!;
const compressFp16 = document.getElementById('compressFp16') as HTMLInputElement;

function showStatus(message: string, type: 'info' | 'success' | 'error') {
  statusDiv.style.display = 'block';
  statusDiv.className = type;
  statusDiv.innerHTML = message;
}

dropzone.addEventListener('click', () => {
  fileInput.click();
});

['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
  dropzone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e: Event) {
  e.preventDefault();
  e.stopPropagation();
}

['dragenter', 'dragover'].forEach((eventName) => {
  dropzone.addEventListener(
    eventName,
    () => {
      dropzone.classList.add('active');
    },
    false,
  );
});

['dragleave', 'drop'].forEach((eventName) => {
  dropzone.addEventListener(
    eventName,
    () => {
      dropzone.classList.remove('active');
    },
    false,
  );
});

dropzone.addEventListener('drop', (e: DragEvent) => {
  const dt = e.dataTransfer;
  if (!dt) return;
  const files = dt.files;
  handleFiles(files);
});

fileInput.addEventListener('change', function (this: HTMLInputElement) {
  if (this.files) {
    handleFiles(this.files);
  }
});

async function handleFiles(files: FileList) {
  if (files.length === 0) return;
  const file = files[0];
  if (!file.name.endsWith('.onnx')) {
    showStatus('Please drop an .onnx file.', 'error');
    return;
  }

  try {
    showStatus(`Loading ${file.name}... (Phase 1/3)`, 'info');
    const arrayBuffer = await file.arrayBuffer();
    const buffer = new Uint8Array(arrayBuffer);

    showStatus(`Parsing ONNX Graph... (Phase 2/3)`, 'info');
    const graph = await load(buffer);

    showStatus(`Compiling to OpenVINO XML/BIN... (Phase 3/3)`, 'info');
    // Let UI update
    await new Promise((r) => setTimeout(r, 50));

    const exporter = new OpenVinoExporter(graph, {
      compressToFp16: compressFp16.checked,
    });

    const { xml, bin } = exporter.export();

    // Use JSZip
    showStatus(`Generating ZIP file...`, 'info');
    const zip = new JSZip();
    const baseName = file.name.replace('.onnx', '');
    zip.file(`${baseName}.xml`, xml);
    zip.file(`${baseName}.bin`, bin);
    zip.file(`${baseName}.mapping`, '<?xml version="1.0" ?>\n<mapping>\n</mapping>');

    const blob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `${baseName}_openvino.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    const originalSizeMB = (file.size / 1024 / 1024).toFixed(2);
    const newSizeMB = (blob.size / 1024 / 1024).toFixed(2);
    showStatus(
      `Success! Downloaded ${baseName}_openvino.zip<br>Original: ${originalSizeMB}MB -> OpenVINO: ${newSizeMB}MB`,
      'success',
    );
  } catch (_err) {
    const err = _err instanceof Error ? _err : new Error(String(_err));
    console.error(err);
    showStatus(`Error compiling model: ${err.message}`, 'error');
  }
}
