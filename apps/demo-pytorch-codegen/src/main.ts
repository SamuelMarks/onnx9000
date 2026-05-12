/* eslint-disable */
import { load, ONNXToPyTorchVisitor } from '@onnx9000/core';

const dropZone = document.getElementById('drop-zone') as HTMLElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const codeArea = document.getElementById('code') as HTMLTextAreaElement;

dropZone.addEventListener('click', () => fileInput.click());

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
    codeArea.value = '# Error: Please provide a valid .onnx file.';
    return;
  }
  codeArea.value = '# Loading and parsing ONNX AST...';
  try {
    const arrayBuffer = await file.arrayBuffer();
    const graph = await load(arrayBuffer);
    const visitor = new ONNXToPyTorchVisitor(graph);
    const code = visitor.generate();
    codeArea.value = code;
  } catch (_err) {
    const err = _err instanceof Error ? _err : new Error(String(_err));
    codeArea.value = `# Error during processing:\n${err.message || err.toString()}`;
  }
}
