/**
 * @fileoverview PyTorch Codegen Demo UI.
 */
import { load, ONNXToPyTorchVisitor } from '@onnx9000/core';

/**
 * Initializes the PyTorch codegen demo UI.
 */
export function initPytorchCodegenDemo(): void {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input') as HTMLInputElement | null;
  const codeArea = document.getElementById('code') as HTMLTextAreaElement | null;

  if (!dropZone || !fileInput || !codeArea) return;

  const validCodeArea = codeArea;

  dropZone.addEventListener('click', () => fileInput.click());

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
      validCodeArea.value = '# Error: Please provide a valid .onnx file.';
      return;
    }
    validCodeArea.value = '# Loading and parsing ONNX AST...';
    try {
      const arrayBuffer = await file.arrayBuffer();
      const graph = await load(arrayBuffer);
      const visitor = new ONNXToPyTorchVisitor(graph);
      const code = visitor.generate();
      validCodeArea.value = code;
    } catch (_err) {
      const err = _err instanceof Error ? _err : new Error(String(_err));
      validCodeArea.value = `# Error during processing:
${err.message}`;
    }
  }
}

initPytorchCodegenDemo();
