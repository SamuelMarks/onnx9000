import { describe, it, expect, vi } from 'vitest';
import { initPytorchCodegenDemo } from '../src/main.js';

vi.mock('@onnx9000/core', () => ({
  load: vi.fn().mockResolvedValue({}),
  ONNXToPyTorchVisitor: class {
    constructor() {}
    generate() {
      return 'import torch\n';
    }
  },
}));

describe('demo-pytorch-codegen', () => {
  it('should generate code', async () => {
    document.body.innerHTML = `
      <div id="drop-zone"></div>
      <input id="file-input" type="file" />
      <textarea id="code"></textarea>
    `;
    initPytorchCodegenDemo();

    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    Object.defineProperty(fileInput, 'files', { value: [new File([''], 'model.onnx')] });
    fileInput.dispatchEvent(new Event('change'));

    await new Promise((r) => setTimeout(r, 10));
    expect((document.getElementById('code') as HTMLTextAreaElement).value).toContain(
      'import torch',
    );
  });
});
